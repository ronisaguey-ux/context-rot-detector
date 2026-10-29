#!/usr/bin/env python3
"""
End-to-End Orchestrator (run_pipeline.py) [V2 Architecture-Specific]

Executes Phases 1 through 5 sequentially:
  Phase 1: Ingestion (SQLite + AGENTS.md -> data/steps.duckdb with tokens_since_compaction & cache telemetry)
  Phase 2: Signal Extraction (Signals 1-15 + compaction_events -> outputs/step_signals.csv & outputs/compaction_events.csv)
  Phase 3: Scoring & Bucketing (Architecture weights W=14.8, >=2 hard-failure floor -> outputs/bucket_scores.csv)
  Phase 4: Analysis, Compaction Recovery Knee, 14c & Task-Type Stratification (-> outputs/score_curve.csv, outputs/report.html)
  Phase 5: Validation Suite (-> outputs/validation_results.json)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from analyze import run_analysis
from generate_sample_db import generate_sample_opencode_db
from ingest import TOKENIZER_BACKEND, run_ingestion
from score import run_scoring
from signals import run_signal_extraction
from validate import print_validation_table, run_validation


# Every stage writes to paths relative to the project ("data/…", "outputs/…"), so a cron
# or systemd invocation from / would scatter artefacts across the filesystem. Anchor the
# whole run to the directory this file lives in; the flags below still resolve relative
# paths against it, and -C/--chdir keeps an interactive caller's own cwd working.
PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete Context Rot Detector V2 pipeline (Phases 1-5).")
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/sample_opencode.db",
        help="Path to OpenCode SQLite database (if missing and default is used, generates 900k sample DB)",
    )
    parser.add_argument("--duckdb", type=str, default="data/steps.duckdb", help="Path to DuckDB store")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--agents-md", type=str, default="AGENTS.md", help="Path to AGENTS.md rules file")
    parser.add_argument("--min-fill-tokens", type=int, default=None, help="Override --min-fill-tokens guard")
    parser.add_argument("--bootstrap", action="store_true", help="Use bootstrap priors for signal baselines")
    parser.add_argument("--keep-cwd", action="store_true", help="Do not cd into the project root")
    args = parser.parse_args()
    if not args.keep_cwd:
        os.chdir(PROJECT_ROOT)

    db_path = Path(args.db_path).expanduser()
    if not db_path.exists():
        print(f"[Pipeline V2] SQLite DB not found at {db_path}; generating 900k-window multi-compaction sample DB...")
        generate_sample_opencode_db(db_path)

    print(f"[Phase 1] Ingesting OpenCode SQLite DB ({db_path}) + AGENTS.md ({args.agents_md}) via {TOKENIZER_BACKEND}...")
    steps_df = run_ingestion(db_path, args.duckdb, args.config, args.agents_md)
    max_since = int(steps_df["tokens_since_compaction"].max()) if not steps_df.empty else 0
    max_cum = int(steps_df["cumulative_context_tokens"].max()) if not steps_df.empty else 0
    print(
        f"          -> {len(steps_df)} steps across {steps_df['session_id'].nunique()} sessions "
        f"(max tokens_since_compaction={max_since:,}, max cumulative={max_cum:,})"
    )

    print("[Phase 2] Extracting Signals 1-15 (including s13 AGENTS.md, s14a/b/c Memory Gate, s15 Recovery Ratio)...")
    sig_df, comp_df = run_signal_extraction(
        args.duckdb,
        args.config,
        "outputs/step_signals.csv",
        "outputs/compaction_events.csv",
    )
    non_warmup = sig_df[~sig_df["warmup"].astype(bool)]
    print(
        f"          -> {len(sig_df)} assistant steps ({len(non_warmup)} non-warmup) & "
        f"{len(comp_df)} compaction events saved"
    )

    print("[Phase 3] Scoring 5k tokens_since_compaction buckets (W=14.8, hard failure floor >=2 fires)...")
    bucket_df, sig_rates_df, meta = run_scoring(
        args.duckdb,
        args.config,
        "outputs/bucket_scores.csv",
        "outputs/signal_rates.csv",
        bootstrap=args.bootstrap,
    )
    print(f"          -> {len(bucket_df)} session-epoch buckets scored across {len(sig_rates_df)} 5k windows")

    print("[Phase 5] Running Validation Suite...")
    val_summary = run_validation(
        duckdb_path=args.duckdb,
        sqlite_path=db_path,
        config_path=args.config,
        out_json="outputs/validation_results.json",
    )

    print("[Phase 4] Computing n-Weighted Rot Curve, Compaction Recovery Knee, 14c Stratification & HTML Report...")
    analysis_res = run_analysis(
        duckdb_path=args.duckdb,
        config_path=args.config,
        score_curve_csv="outputs/score_curve.csv",
        stratified_curve_csv="outputs/score_curve_stratified.csv",
        task_type_csv="outputs/task_type_curves.csv",
        report_html="outputs/report.html",
        template_html="report/template.html",
        bootstrap=args.bootstrap,
        min_fill_tokens=args.min_fill_tokens,
        validation_summary=val_summary,
    )
    onset = analysis_res["onset"]
    print(
        f"          -> Recommended Compaction Trigger T = {onset['recommended_compaction_tokens']:,} tokens "
        f"(Bucket #{onset['recommended_bucket']}, binding={onset['binding_method']})\n"
        f"          -> OpenCode V2 compaction.buffer = {onset['opencode_v2_buffer_tokens']:,} tokens "
        f"(900,000 window - {onset['recommended_compaction_tokens']:,} T)"
    )
    print_validation_table(val_summary)


if __name__ == "__main__":
    main()
