#!/usr/bin/env python3
"""
Phase 3 — Scoring & Bucketing (score.py) [V2 Architecture-Specific]

Key upgrades:
  1. Buckets strictly on `tokens_since_compaction` per `(session_id, compaction_index, bucket_id)`
     so pre-compaction and post-compaction epochs never mix.
  2. Uses the architecture-specific 15-signal weight set (`s1..s4=1.3`, `s9..s10=0.3`,
     `s13=1.5`, `s14a=0.5`, `s14b=0.5`, `s14c=1.2`, `s15=1.3`; total W = 14.8).
  3. Requires `(s1_fires + s2_fires) >= hard_failure_min_fires` (default 2) before
     applying the 500 hard-failure floor, preventing single transient tool errors
     from pegging buckets at 500.
  4. Computes per-bucket `confidence` (`min(1.0, n_steps / 10)`) and flags `n_steps < 10`
     as `low_confidence = True`.
  5. Preserves `post_compaction`, `s14c_epoch_lost`, `task_type`, and real prompt-cache
     hit/miss token totals per bucket for Phase 4 analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from ingest import open_duckdb
from signals import ALL_SIGNAL_COLUMNS


DEFAULT_WEIGHTS: Dict[str, float] = {
    "s1": 1.3, "s2": 1.3, "s3": 1.3, "s4": 1.3,
    "s5": 0.8, "s6": 0.8,
    "s7": 0.6, "s8": 0.6,
    "s9": 0.3, "s10": 0.3,
    "s11": 0.6,
    "s12": 0.3,
    "s13": 1.5,
    "s14a": 0.5, "s14b": 0.5, "s14c": 1.2,
    "s15": 1.3,
}

DEFAULT_BOOTSTRAP_PRIORS: Dict[str, float] = {
    "s1": 0.14, "s2": 0.09, "s3": 0.07, "s4": 0.16,
    "s5": 0.11, "s6": 0.10, "s7": 0.06, "s8": 0.08,
    "s9": 0.04, "s10": 0.04, "s11": 0.07, "s12": 0.05,
    "s13": 0.08, "s14a": 0.07, "s14b": 0.06, "s14c": 0.09,
    "s15": 0.08,
}


def load_config(config_path: Optional[str | Path] = "config.yaml") -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "model_context_window": 900000,
        "bucket_size_tokens": 5000,
        "min_steps_per_bucket": 10,
        "baseline_percentile": 95,
        "hard_failure_min_fires": 2,
        "score_floor_on_hard_failure": 500,
        "smoothing_window": 3,
        "min_fill_tokens": 100000,
        "min_fill_fraction_of_max": 0.20,
        "onset_slope_multiplier": 2.0,
        "onset_sustained_buckets": 2,
        "onset_p90_threshold": 400,
        "per_signal_onset_multiplier": 1.5,
        "recovery_ratio_threshold": 0.65,
        "price_cached_input_per_mtok": 0.028,
        "price_uncached_input_per_mtok": 0.280,
        "price_output_per_mtok": 1.100,
        "weights": DEFAULT_WEIGHTS.copy(),
        "bootstrap_priors": DEFAULT_BOOTSTRAP_PRIORS.copy(),
    }
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
            for k, v in loaded.items():
                if k in ("weights", "bootstrap_priors") and isinstance(v, dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v
    return cfg


def compute_baselines(
    bucket_df: pd.DataFrame,
    percentile: float = 95.0,
    bootstrap: bool = False,
    bootstrap_priors: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], bool]:
    priors = {**DEFAULT_BOOTSTRAP_PRIORS, **(bootstrap_priors or {})}
    n_sessions = bucket_df["session_id"].nunique() if not bucket_df.empty else 0
    confident_df = bucket_df[~bucket_df["low_confidence"].astype(bool)] if "low_confidence" in bucket_df.columns else bucket_df
    if confident_df.empty:
        confident_df = bucket_df

    used_bootstrap = bool(bootstrap or n_sessions < 10 or confident_df.empty)
    baselines: Dict[str, float] = {}

    for sig in ALL_SIGNAL_COLUMNS:
        rate_col = f"{sig}_rate"
        prior_val = float(priors.get(sig, 0.08))

        if bootstrap or confident_df.empty or rate_col not in confident_df.columns:
            baselines[sig] = prior_val
            continue

        rates = confident_df[rate_col].dropna().to_numpy(dtype=float)
        if len(rates) == 0:
            baselines[sig] = prior_val
            continue

        emp_p95 = float(np.percentile(rates, percentile))
        if n_sessions < 10:
            alpha = n_sessions / 10.0
            blended = alpha * emp_p95 + (1.0 - alpha) * prior_val
            baselines[sig] = max(blended, prior_val * 0.5, 1e-4)
        else:
            baselines[sig] = emp_p95 if emp_p95 > 1e-6 else prior_val

    return baselines, used_bootstrap


def score_buckets(
    step_signals: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    bootstrap: bool = False,
    return_metadata: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, Any]]:
    cfg = config or load_config()
    bucket_size = int(cfg.get("bucket_size_tokens", 5000))
    min_steps = int(cfg.get("min_steps_per_bucket", 10))
    percentile = float(cfg.get("baseline_percentile", 95))
    hard_min_fires = int(cfg.get("hard_failure_min_fires", 2))
    hard_floor = int(cfg.get("score_floor_on_hard_failure", 500))
    smoothing_window = int(cfg.get("smoothing_window", 3))

    weights: Dict[str, float] = {
        sig: float(cfg["weights"].get(sig, DEFAULT_WEIGHTS[sig]))
        for sig in ALL_SIGNAL_COLUMNS
    }
    total_weight_w = sum(weights.values())
    priors = cfg.get("bootstrap_priors", DEFAULT_BOOTSTRAP_PRIORS)

    # Ensure backward-compatible columns if called from synthetic minimal test DataFrames
    df_in = step_signals.copy()
    for col, default_val in [
        ("task_type", "feature"),
        ("compaction_index", 0),
        ("post_compaction", False),
        ("tokens_since_compaction", 0),
        ("s14c_epoch_lost", False),
        ("prompt_cache_hit_tokens", 0),
        ("prompt_cache_miss_tokens", 0),
        ("has_real_cache_telemetry", False),
        ("tokens_output", 0),
    ]:
        if col not in df_in.columns:
            df_in[col] = default_val
    for sig in ALL_SIGNAL_COLUMNS:
        if sig not in df_in.columns:
            df_in[sig] = False

    non_warmup = df_in[~df_in["warmup"].astype(bool)].copy() if not df_in.empty else df_in.copy()

    if non_warmup.empty:
        empty_cols = [
            "session_id", "task_type", "compaction_index", "post_compaction", "s14c_epoch_lost",
            "bucket_id", "tokens_since_compaction_k", "n_steps", "confidence", "low_confidence",
            "prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "tokens_output", "has_real_cache_telemetry",
            "score_raw", "score_smooth",
            *[f"{sig}_fires" for sig in ALL_SIGNAL_COLUMNS],
            *[f"{sig}_rate" for sig in ALL_SIGNAL_COLUMNS],
            *[f"{sig}_norm" for sig in ALL_SIGNAL_COLUMNS],
        ]
        empty_df = pd.DataFrame(columns=empty_cols)
        if return_metadata:
            return empty_df, {"baselines": priors, "used_bootstrap": True, "total_weight_w": total_weight_w}
        return empty_df

    # Group by (session_id, compaction_index, bucket_id) so pre- and post-compaction epochs are distinct
    agg_records: List[Dict[str, Any]] = []
    for (sess_id, comp_idx, b_id), grp in non_warmup.groupby(["session_id", "compaction_index", "bucket_id"], sort=True):
        n_steps = len(grp)
        tokens_k = int(b_id) * (bucket_size // 1000)

        rec: Dict[str, Any] = {
            "session_id": str(sess_id),
            "task_type": str(grp["task_type"].iloc[0]),
            "compaction_index": int(comp_idx),
            "post_compaction": bool(int(comp_idx) > 0 or grp["post_compaction"].astype(bool).any()),
            "s14c_epoch_lost": bool(grp["s14c_epoch_lost"].astype(bool).any()),
            "bucket_id": int(b_id),
            "tokens_since_compaction_k": tokens_k,
            "context_tokens_k": tokens_k,  # Alias for compatibility
            "n_steps": int(n_steps),
            "confidence": 1.0,
            "low_confidence": False,
            "prompt_cache_hit_tokens": int(grp["prompt_cache_hit_tokens"].sum()),
            "prompt_cache_miss_tokens": int(grp["prompt_cache_miss_tokens"].sum()),
            "tokens_output": int(grp["tokens_output"].sum()),
            "has_real_cache_telemetry": bool(grp["has_real_cache_telemetry"].astype(bool).any()),
        }
        for sig in ALL_SIGNAL_COLUMNS:
            fires = int(grp[sig].astype(bool).sum())
            rec[f"{sig}_fires"] = fires
            rec[f"{sig}_rate"] = float(fires / n_steps) if n_steps > 0 else 0.0
        agg_records.append(rec)

    bucket_df = (
        pd.DataFrame(agg_records)
        .sort_values(["session_id", "compaction_index", "bucket_id"])
        .reset_index(drop=True)
    )
    # Item 11: Compute per-bucket confidence from total steps in bucket across sessions (flag n < min_steps_per_bucket as low_confidence)
    bucket_step_totals = bucket_df.groupby("bucket_id")["n_steps"].transform("sum")
    bucket_df["bucket_total_steps"] = bucket_step_totals.astype(int)
    bucket_df["confidence"] = np.minimum(1.0, bucket_step_totals.astype(float) / float(max(1, min_steps))).round(3)
    bucket_df["low_confidence"] = bucket_step_totals < min_steps

    baselines, used_bootstrap = compute_baselines(
        bucket_df,
        percentile=percentile,
        bootstrap=bootstrap,
        bootstrap_priors=priors,
    )

    for sig in ALL_SIGNAL_COLUMNS:
        base_val = max(float(baselines[sig]), 1e-6)
        bucket_df[f"{sig}_norm"] = np.minimum(1.0, bucket_df[f"{sig}_rate"] / base_val)

    # Composite score -> 1-1000
    weighted_sum = np.zeros(len(bucket_df), dtype=float)
    for sig in ALL_SIGNAL_COLUMNS:
        weighted_sum += weights[sig] * bucket_df[f"{sig}_norm"].to_numpy(dtype=float)

    weighted_0_1 = weighted_sum / max(total_weight_w, 1e-9)
    raw_scores = np.rint(1.0 + weighted_0_1 * 999.0).astype(int)
    raw_scores = np.clip(raw_scores, 1, 1000)

    # Item 4: Require >= hard_failure_min_fires (default 2) hard failures in the bucket before flooring at 500
    hard_failure_mask = (bucket_df["s1_fires"] + bucket_df["s2_fires"]) >= hard_min_fires
    raw_scores = np.where(hard_failure_mask, np.maximum(raw_scores, hard_floor), raw_scores)
    bucket_df["score_raw"] = raw_scores

    # Median-of-3 smoothing within each (session_id, compaction_index) trajectory
    smooth_scores = np.zeros(len(bucket_df), dtype=int)
    for (_, _), idxs in bucket_df.groupby(["session_id", "compaction_index"], sort=False).groups.items():
        idx_list = list(idxs)
        sess_raw = bucket_df.loc[idx_list, "score_raw"].to_numpy(dtype=float)
        n_b = len(sess_raw)
        sess_smooth = np.zeros(n_b, dtype=int)
        half_w = smoothing_window // 2
        for j in range(n_b):
            lo = max(0, j - half_w)
            hi = min(n_b, j + half_w + 1)
            sess_smooth[j] = int(round(float(np.median(sess_raw[lo:hi]))))
        smooth_scores[idx_list] = sess_smooth

    bucket_df["score_smooth"] = smooth_scores

    if return_metadata:
        meta = {
            "baselines": baselines,
            "used_bootstrap": used_bootstrap,
            "total_weight_w": round(total_weight_w, 4),
            "weights": weights,
            "hard_failure_min_fires": hard_min_fires,
        }
        return bucket_df, meta
    return bucket_df


def aggregate_signal_rates_by_bucket(
    bucket_df: pd.DataFrame,
    baselines: Dict[str, float],
) -> pd.DataFrame:
    """
    Produces `outputs/signal_rates.csv`: n_steps-weighted per-bucket signal firing rates
    across all sessions.
    """
    if bucket_df.empty:
        return pd.DataFrame()

    confident = bucket_df[~bucket_df["low_confidence"].astype(bool)]
    if confident.empty:
        confident = bucket_df

    rows = []
    for b_id, grp in confident.groupby("bucket_id", sort=True):
        w_steps = grp["n_steps"].to_numpy(dtype=float)
        total_steps = int(w_steps.sum())
        r: Dict[str, Any] = {
            "bucket_id": int(b_id),
            "tokens_since_compaction_k": int(grp["tokens_since_compaction_k"].iloc[0]),
            "context_tokens_k": int(grp["tokens_since_compaction_k"].iloc[0]),
            "n_sessions": int(grp["session_id"].nunique()),
            "total_steps": total_steps,
        }
        for sig in ALL_SIGNAL_COLUMNS:
            rates = grp[f"{sig}_rate"].to_numpy(dtype=float)
            weighted_mean_rate = float(np.sum(rates * w_steps) / max(1.0, float(total_steps)))
            base_val = float(baselines.get(sig, 0.08))
            r[f"{sig}_mean_rate"] = round(weighted_mean_rate, 5)
            r[f"{sig}_baseline"] = round(base_val, 5)
            r[f"{sig}_ratio_to_baseline"] = round(weighted_mean_rate / max(base_val, 1e-6), 3)
        rows.append(r)

    return pd.DataFrame(rows)


def run_scoring(
    duckdb_path: str | Path = "data/steps.duckdb",
    config_path: Optional[str | Path] = "config.yaml",
    bucket_scores_csv: str | Path = "outputs/bucket_scores.csv",
    signal_rates_csv: str | Path = "outputs/signal_rates.csv",
    bootstrap: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    cfg = load_config(config_path)
    with open_duckdb(duckdb_path, read_only=True) as con:
        step_signals = con.execute("SELECT * FROM step_signals ORDER BY session_id, step_index").df()

    bucket_df, meta = score_buckets(step_signals, config=cfg, bootstrap=bootstrap, return_metadata=True)  # type: ignore
    sig_rates_df = aggregate_signal_rates_by_bucket(bucket_df, meta["baselines"])

    with open_duckdb(duckdb_path, read_only=False) as con:
        con.execute("DROP TABLE IF EXISTS bucket_scores")
        con.register("bucket_df_view", bucket_df)
        con.execute("CREATE TABLE bucket_scores AS SELECT * FROM bucket_df_view")

    Path(bucket_scores_csv).parent.mkdir(parents=True, exist_ok=True)
    bucket_df.to_csv(bucket_scores_csv, index=False)

    Path(signal_rates_csv).parent.mkdir(parents=True, exist_ok=True)
    sig_rates_df.to_csv(signal_rates_csv, index=False)

    return bucket_df, sig_rates_df, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: Compute 1-1000 Context Rot scores per 5k-token bucket.")
    parser.add_argument("--duckdb", type=str, default="data/steps.duckdb", help="Path to steps.duckdb")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--out-buckets", type=str, default="outputs/bucket_scores.csv", help="Output bucket_scores.csv")
    parser.add_argument("--out-rates", type=str, default="outputs/signal_rates.csv", help="Output signal_rates.csv")
    parser.add_argument("--bootstrap", action="store_true", help="Use prior baselines for cold-start sessions")
    args = parser.parse_args()

    bucket_df, sig_rates_df, meta = run_scoring(
        args.duckdb, args.config, args.out_buckets, args.out_rates, bootstrap=args.bootstrap
    )
    print(f"[Phase 3: Scoring] Scored {len(bucket_df)} session-buckets across {len(sig_rates_df)} unique 5k-token windows.")
    print(
        f"[Phase 3: Scoring] Bootstrap={meta['used_bootstrap']} | Total weight W={meta['total_weight_w']} | "
        f"Hard failure floor requires >={meta['hard_failure_min_fires']} fires"
    )


if __name__ == "__main__":
    main()
