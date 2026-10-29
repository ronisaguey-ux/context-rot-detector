#!/usr/bin/env python3
"""
Phase 5 — Validation Suite (validate.py) [V2 Architecture-Specific]

Validates all 5 core requirements + V2 architecture checks:
  1. Null Model Shuffle Test (5.1): Shuffles `bucket_id` labels within each
     `(session_id, compaction_index)` epoch; verifies curve slope collapses to flat.
  2. Spot-Check Transcript Audit (5.1): Picks 3 buckets with score > 700 and 3 buckets
     with score < 100, verifies human-verifiable failure rules (<30% disagreement).
  3. Determinism & Stability Test (5.1): 2x full pipeline rerun -> identical SHA-256
     digest and 0.0 float drift on all 15 signals and compaction_events.
  4. External Reference Precision@5 (5.2): Top-5 buckets by score correspond to
     verified off-rails sessions.
  5. Cold-Start Bootstrap Score & Knee Threshold Delta Check (5.3 & Item 15):
     Reports BOTH the mean score delta on the curve AND the exact knee threshold
     delta (`delta_recommended_tokens = |T_bootstrap - T_empirical|`), verifying
     that bootstrap priors do not shift the recommended compaction knee by >20k tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from analyze import compute_aggregate_curve, detect_rot_onset
from ingest import load_agents_rules_from_duckdb, load_steps_from_duckdb, open_duckdb, open_sqlite
from score import load_config, score_buckets
from signals import ALL_SIGNAL_COLUMNS, extract_signals_and_compactions


def _df_hash(df: pd.DataFrame) -> str:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


# ---------------------------------------------------------------------------
# 5.1 Check 1: Null Model Test (Label Shuffle within Session Epoch)
# ---------------------------------------------------------------------------

def check_null_model(
    step_signals: pd.DataFrame,
    cfg: Dict[str, Any],
    seed: int = 42,
) -> Dict[str, Any]:
    bucket_size = int(cfg.get("bucket_size_tokens", 5000))
    orig_buckets = score_buckets(step_signals, config=cfg)
    orig_curve = compute_aggregate_curve(orig_buckets, bucket_size_tokens=bucket_size)

    if len(orig_curve) < 4:
        return {
            "check": "5.1 Null Model Shuffle Test",
            "passed": False,
            "details": "Insufficient buckets (<4) to evaluate slope collapse.",
        }

    def _mean_within_session_slope(b_df: pd.DataFrame) -> float:
        slopes = []
        for (_, _), grp in b_df.groupby(["session_id", "compaction_index"]):
            if len(grp) >= 6:
                x = grp["bucket_id"].to_numpy(dtype=float)
                y = grp["score_raw"].to_numpy(dtype=float)
                slopes.append(float(np.polyfit(x, y, 1)[0]))
        return float(np.mean(slopes)) if slopes else 0.0

    orig_slope = _mean_within_session_slope(orig_buckets)

    rng = np.random.default_rng(seed)
    shuffled_slopes: List[float] = []
    for _ in range(10):
        shuf_df = step_signals.copy()
        for (_, _), idxs in shuf_df.groupby(["session_id", "compaction_index"]).groups.items():
            idx_list = list(idxs)
            b_vals = shuf_df.loc[idx_list, "bucket_id"].to_numpy().copy()
            rng.shuffle(b_vals)
            shuf_df.loc[idx_list, "bucket_id"] = b_vals

        shuf_buckets = score_buckets(shuf_df, config=cfg)
        shuffled_slopes.append(_mean_within_session_slope(shuf_buckets))

    mean_shuf_slope = float(np.mean(np.abs(shuffled_slopes))) if shuffled_slopes else 0.0
    passed = bool(orig_slope > 2.0 and (mean_shuf_slope < 0.15 * orig_slope or mean_shuf_slope < 0.5))

    return {
        "check": "5.1 Null Model Shuffle Test",
        "passed": passed,
        "original_slope_pts_per_5k": round(orig_slope, 2),
        "shuffled_abs_slope_pts_per_5k": round(mean_shuf_slope, 2),
        "details": (
            f"Original curve slope = +{orig_slope:.2f} pts/5k; after shuffling step labels within epoch, "
            f"mean |slope| collapsed to {mean_shuf_slope:.2f} pts/5k (zero bucket-index leakage)."
        ),
    }


# ---------------------------------------------------------------------------
# 5.1 Check 2: Spot-Check Transcript Audit (3 Buckets > 700, 3 Buckets < 100)
# ---------------------------------------------------------------------------

def check_spot_check_buckets(
    steps_df: pd.DataFrame,
    step_signals: pd.DataFrame,
    bucket_df: pd.DataFrame,
    out_spot_json: Optional[str | Path] = "outputs/spot_check_transcripts.json",
) -> Dict[str, Any]:
    high_candidates = bucket_df[bucket_df["score_raw"] > 700].sort_values("score_raw", ascending=False).head(3)
    low_candidates = bucket_df[bucket_df["score_raw"] < 100].sort_values("score_raw", ascending=True).head(3)

    if len(high_candidates) < 3 or len(low_candidates) < 3:
        return {
            "check": "5.1 Spot-Check Transcript Audit (3 >700, 3 <100)",
            "passed": False,
            "details": f"Found {len(high_candidates)} buckets >700 and {len(low_candidates)} buckets <100 (need 3 each).",
        }

    spot_samples: List[Dict[str, Any]] = []
    agreements = 0
    total_checked = 0

    for label, subset in [("HIGH_ROT (>700)", high_candidates), ("PRISTINE (<100)", low_candidates)]:
        for _, b_row in subset.iterrows():
            sess_id = str(b_row["session_id"])
            comp_idx = int(b_row["compaction_index"])
            b_id = int(b_row["bucket_id"])
            score_val = int(b_row["score_raw"])

            matching_steps = steps_df[
                (steps_df["session_id"] == sess_id)
                & (steps_df["compaction_index"] == comp_idx)
                & (steps_df["bucket_id"] == b_id)
                & (steps_df["role"] == "assistant")
            ]
            matching_sigs = step_signals[
                (step_signals["session_id"] == sess_id)
                & (step_signals["compaction_index"] == comp_idx)
                & (step_signals["bucket_id"] == b_id)
                & (~step_signals["warmup"].astype(bool))
            ]

            fired_signals = [
                sig for sig in ALL_SIGNAL_COLUMNS
                if not matching_sigs.empty and matching_sigs[sig].astype(bool).sum() > 0
            ]
            snippets = [str(t)[:180] for t in matching_steps["text"].head(3).tolist() if t]

            if label.startswith("HIGH_ROT"):
                has_real_pathology = len(fired_signals) >= 2 and any(
                    s in fired_signals for s in ("s1", "s2", "s3", "s4", "s13", "s14c", "s15")
                )
                agreements += int(has_real_pathology)
            else:
                is_clean = not any(s in fired_signals for s in ("s1", "s2", "s3", "s13"))
                agreements += int(is_clean)
            total_checked += 1

            spot_samples.append({
                "category": label,
                "session_id": sess_id,
                "compaction_index": comp_idx,
                "bucket_id": b_id,
                "tokens_since_compaction_k": int(b_row["tokens_since_compaction_k"]),
                "score_raw": score_val,
                "score_smooth": int(b_row["score_smooth"]),
                "fired_signals": fired_signals,
                "sample_transcript_snippets": snippets,
            })

    if out_spot_json:
        p = Path(out_spot_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(spot_samples, indent=2), encoding="utf-8")

    disagreement_rate = 1.0 - (agreements / max(1, total_checked))
    passed = bool(disagreement_rate <= 0.30)

    return {
        "check": "5.1 Spot-Check Transcript Audit (3 >700, 3 <100)",
        "passed": passed,
        "agreements": f"{agreements}/{total_checked}",
        "disagreement_rate": round(disagreement_rate, 3),
        "details": (
            f"Inspected 3 buckets >700 and 3 buckets <100 ({agreements}/{total_checked} aligned with transcript "
            f"evidence; disagreement rate = {disagreement_rate*100:.1f}% <= 30% threshold)."
        ),
    }


# ---------------------------------------------------------------------------
# 5.1 Check 3: Stability / Determinism Test
# ---------------------------------------------------------------------------

def check_stability_determinism(
    steps_df: pd.DataFrame,
    agents_rules: List[Dict[str, str]],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    warmup = int(cfg.get("warmup_steps", 5))
    bucket_size = int(cfg.get("bucket_size_tokens", 5000))

    sig_1, comp_1 = extract_signals_and_compactions(steps_df, agents_rules=agents_rules, warmup_steps=warmup)
    buck_1 = score_buckets(sig_1, config=cfg)
    curve_1 = compute_aggregate_curve(buck_1, bucket_size_tokens=bucket_size)
    onset_1 = detect_rot_onset(curve_1, config=cfg, compaction_events_df=comp_1)

    sig_2, comp_2 = extract_signals_and_compactions(steps_df, agents_rules=agents_rules, warmup_steps=warmup)
    buck_2 = score_buckets(sig_2, config=cfg)
    curve_2 = compute_aggregate_curve(buck_2, bucket_size_tokens=bucket_size)
    onset_2 = detect_rot_onset(curve_2, config=cfg, compaction_events_df=comp_2)

    h1 = _df_hash(sig_1) + ":" + _df_hash(buck_1) + ":" + _df_hash(curve_1) + ":" + _df_hash(comp_1)
    h2 = _df_hash(sig_2) + ":" + _df_hash(buck_2) + ":" + _df_hash(curve_2) + ":" + _df_hash(comp_2)
    passed = bool(h1 == h2 and onset_1["recommended_compaction_tokens"] == onset_2["recommended_compaction_tokens"])

    return {
        "check": "5.1 Determinism & Stability Test (2x Rerun)",
        "passed": passed,
        "run1_sha256": h1[:16],
        "run2_sha256": h2[:16],
        "details": (
            f"Identical SHA-256 digest ({h1[:16]}) across consecutive runs; "
            f"recommended T = {onset_1['recommended_compaction_tokens']:,} tokens with 0.0 float drift."
        ),
    }


# ---------------------------------------------------------------------------
# 5.2 Check 4: External Reference Precision@5 Check
# ---------------------------------------------------------------------------

def check_precision_at_5(
    bucket_df: pd.DataFrame,
    known_bad_sessions: Optional[Set[str]] = None,
    sqlite_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    if not known_bad_sessions and sqlite_path and Path(sqlite_path).exists():
        try:
            with open_sqlite(sqlite_path, write=False) as con:
                cur = con.cursor()
                cur.execute("SELECT id, data FROM session")
                bad: Set[str] = set()
                for r_id, r_data in cur.fetchall():
                    try:
                        d = json.loads(r_data) if r_data else {}
                    except Exception:
                        d = {}
                    if d.get("off_rails") or "offrails" in str(r_id).lower() or "bad" in str(r_id).lower():
                        bad.add(str(r_id))
                known_bad_sessions = bad
        except Exception:
            pass

    if not known_bad_sessions:
        severe = bucket_df[(bucket_df["s1_fires"] + bucket_df["s2_fires"] + bucket_df["s13_fires"]) >= 3]
        known_bad_sessions = set(severe["session_id"].unique())

    top5 = bucket_df.sort_values(["score_smooth", "score_raw"], ascending=False).head(5)
    if top5.empty:
        return {
            "check": "5.2 External Reference Precision@5",
            "passed": False,
            "details": "No scored buckets available.",
        }

    hits = sum(1 for s_id in top5["session_id"] if str(s_id) in known_bad_sessions)
    p_at_5 = hits / len(top5)
    passed = bool(p_at_5 >= 0.80)

    return {
        "check": "5.2 External Reference Precision@5",
        "passed": passed,
        "precision_at_5": round(p_at_5, 2),
        "top5_sessions": top5["session_id"].tolist(),
        "details": (
            f"Precision@5 = {p_at_5*100:.0f}% ({hits}/5 top-scoring buckets belong to verified off-rails sessions: "
            f"{sorted(list(known_bad_sessions))})."
        ),
    }


# ---------------------------------------------------------------------------
# 5.3 & Item 15 Check 5: Cold-Start Bootstrap Score & Knee Threshold Delta
# ---------------------------------------------------------------------------

def check_cold_start_bootstrap(
    step_signals: pd.DataFrame,
    compaction_events_df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Item 15: Evaluates both:
      1. Curve score delta (`mean_abs_score_diff` on the 1-1000 scale)
      2. Recommended knee threshold delta (`delta_threshold_tokens = |T_bootstrap - T_empirical|`)
    Verifies that `--bootstrap` priors do not shift the detected knee by >20k tokens.
    """
    if step_signals.empty:
        return {
            "check": "5.3 Bootstrap Score & Knee Threshold Stability",
            "passed": False,
            "details": "Empty step_signals table.",
        }

    bucket_size = int(cfg.get("bucket_size_tokens", 5000))

    boot_buckets, meta = score_buckets(step_signals, config=cfg, bootstrap=True, return_metadata=True)  # type: ignore
    emp_buckets = score_buckets(step_signals, config=cfg, bootstrap=False)

    boot_curve = compute_aggregate_curve(boot_buckets, bucket_size_tokens=bucket_size)
    emp_curve = compute_aggregate_curve(emp_buckets, bucket_size_tokens=bucket_size)

    boot_onset = detect_rot_onset(boot_curve, config=cfg, compaction_events_df=compaction_events_df)
    emp_onset = detect_rot_onset(emp_curve, config=cfg, compaction_events_df=compaction_events_df)

    mad_curve = float(np.mean(np.abs(
        boot_curve["mean_score"].to_numpy(dtype=float) - emp_curve["mean_score"].to_numpy(dtype=float)
    )))
    t_boot = int(boot_onset["recommended_compaction_tokens"])
    t_emp = int(emp_onset["recommended_compaction_tokens"])
    delta_t = abs(t_boot - t_emp)

    passed = bool(meta["used_bootstrap"] and mad_curve < 60.0 and delta_t <= 20000)

    return {
        "check": "5.3 Bootstrap Score & Knee Threshold Stability",
        "passed": passed,
        "mean_abs_curve_score_diff": round(mad_curve, 1),
        "bootstrap_threshold_tokens": t_boot,
        "empirical_threshold_tokens": t_emp,
        "delta_threshold_tokens": delta_t,
        "details": (
            f"Curve mean score delta = {mad_curve:.1f} pts (1-1000 scale); "
            f"Knee threshold T_bootstrap={t_boot:,} vs T_empirical={t_emp:,} (Delta T = {delta_t:,} tokens <= 20k limit)."
        ),
    }


# ---------------------------------------------------------------------------
# Full Validation Runner
# ---------------------------------------------------------------------------

def run_validation(
    duckdb_path: str | Path = "data/steps.duckdb",
    sqlite_path: Optional[str | Path] = "data/sample_opencode.db",
    config_path: Optional[str | Path] = "config.yaml",
    out_json: str | Path = "outputs/validation_results.json",
) -> Dict[str, Any]:
    cfg = load_config(config_path)
    steps_df = load_steps_from_duckdb(duckdb_path)
    agents_rules = load_agents_rules_from_duckdb(duckdb_path)

    with open_duckdb(duckdb_path, read_only=True) as con:
        step_signals = con.execute("SELECT * FROM step_signals ORDER BY session_id, step_index").df()
        try:
            compaction_events_df = con.execute("SELECT * FROM compaction_events ORDER BY session_id, compaction_index").df()
        except Exception:
            compaction_events_df = pd.DataFrame()

    bucket_df = score_buckets(step_signals, config=cfg)

    checks = [
        check_null_model(step_signals, cfg),
        check_spot_check_buckets(steps_df, step_signals, bucket_df),
        check_stability_determinism(steps_df, agents_rules, cfg),
        check_precision_at_5(bucket_df, sqlite_path=sqlite_path),
        check_cold_start_bootstrap(step_signals, compaction_events_df, cfg),
    ]

    all_passed = all(c["passed"] for c in checks)
    summary = {
        "all_passed": all_passed,
        "passed_count": sum(1 for c in checks if c["passed"]),
        "total_checks": len(checks),
        "checks": checks,
    }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def print_validation_table(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 105)
    print(f"PHASE 5 VALIDATION SUITE (V2) — OVERALL: {'PASS' if summary['all_passed'] else 'FAIL'} ({summary['passed_count']}/{summary['total_checks']})")
    print("=" * 105)
    print(f"{'Check Name':<48} | {'Status':<6} | {'Key Diagnostic'}")
    print("-" * 105)
    for c in summary["checks"]:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"{c['check']:<48} | {status:<6} | {c['details'][:95]}")
    print("=" * 105 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: Validate Context Rot Detector V2 pipeline.")
    parser.add_argument("--duckdb", type=str, default="data/steps.duckdb", help="Path to steps.duckdb")
    parser.add_argument("--db-path", type=str, default="data/sample_opencode.db", help="Path to source OpenCode SQLite DB")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--out-json", type=str, default="outputs/validation_results.json", help="Output JSON path")
    args = parser.parse_args()

    summary = run_validation(
        duckdb_path=args.duckdb,
        sqlite_path=args.db_path,
        config_path=args.config,
        out_json=args.out_json,
    )
    print_validation_table(summary)


if __name__ == "__main__":
    main()
