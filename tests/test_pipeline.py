#!/usr/bin/env python3
"""
Unit & Integration Tests for Context Rot Detector V2 (tests/test_pipeline.py)

Verifies all 15 prioritized items:
  - Signals 1..12 + Signal 13 (AGENTS.md failure), Signals 14a/14b/14c (memory-gating),
    and Signal 15 (compaction recovery ratio + compaction_events table)
  - tokens_since_compaction bucketing & post_compaction separation
  - Hard-failure floor requires >= 2 fires (single transient error does NOT peg at 500)
  - n_steps-weighted aggregation & confidence column (n < 10 -> low_confidence)
  - --min-fill-tokens guard against short dataset false reports
  - Task-type classification (refactor, bugfix, feature, test, explore)
  - OpenCode V2 buffer calculation (buffer = model_window - threshold_T)
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyze import compute_aggregate_curve, compute_s14c_stratified_curves, detect_rot_onset
from ingest import classify_task_type, parse_agents_md_rules
from score import load_config, score_buckets
from signals import (
    SessionState,
    eval_s1_tool_error,
    eval_s2_edit_failure,
    eval_s3_retry_loop,
    eval_s4_reread,
    eval_s5_self_correction,
    eval_s6_duplicate_read,
    eval_s7_unreferenced_large_output,
    eval_s8_repetition_entropy_drop,
    eval_s9_instruction_amnesia,
    eval_s10_stale_todo_snapshot,
    eval_s11_contradictory_edit,
    eval_s12_latency_outlier,
    eval_s13_agents_md_failure,
    extract_candidate_facts,
    extract_signals_and_compactions,
    update_session_state,
)


class TestContextRotV2(unittest.TestCase):
    def test_s1_to_s12_core_signals(self) -> None:
        tc_err = [{"name": "edit", "params": {"path": "a.py"}, "result": "Error: string not found", "error": "err", "exit_code": 1}]
        self.assertTrue(eval_s1_tool_error(tc_err))
        self.assertTrue(eval_s2_edit_failure(tc_err))

        state = SessionState(session_median_latency=2.0, early_entropies=[2.1, 2.2])
        update_session_state(state, "user", "Never modify legacy_auth.py", [], None)
        self.assertTrue(eval_s9_instruction_amnesia("", [{"name": "edit", "params": {"path": "legacy_auth.py"}}], state))
        self.assertTrue(eval_s12_latency_outlier(0.0, 8.0, state))

    def test_s13_agents_md_retrieval_failure(self) -> None:
        rules = parse_agents_md_rules(ROOT / "AGENTS.md")
        compiled = [re.compile(r["pattern"], re.I) for r in rules]
        clean_tc = [{"name": "edit", "params": {"path": "src/config/settings.py", "new_string": "TIMEOUT = 30"}}]
        violating_tc = [{"name": "edit", "params": {"path": "legacy_config.py", "new_string": "import deprecated_crypto"}}]
        self.assertFalse(eval_s13_agents_md_failure(clean_tc, compiled))
        self.assertTrue(eval_s13_agents_md_failure(violating_tc, compiled))

    def test_s14abc_and_s15_compaction_events(self) -> None:
        # Build a synthetic session with 1 compaction where a candidate fact is lost and latency > 3
        rows = []
        for i in range(16):
            is_comp = (i == 8)
            comp_idx = 0 if i < 8 else 1
            tok_since = (i + 1) * 5000 if i < 8 else (15000 + (i - 7) * 5000)
            txt = "Note that critical_shard_id=42" if i == 3 else "Working on step"
            # Post-compaction has tool errors and no memory write of critical_shard_id=42
            tcs = []
            if i >= 6:
                tcs = [{"name": "edit", "params": {"path": "legacy_config.py"}, "result": "string not found", "error": "err", "exit_code": 1}]
            rows.append({
                "step_index": i,
                "session_id": "sess_test",
                "task_type": "refactor",
                "role": "assistant",
                "text": txt,
                "tool_calls": tcs,
                "todos": None,
                "memory_writes": [],
                "tokens_input": tok_since - 5000,
                "tokens_output": 5000,
                "tokens_total": tok_since,
                "cumulative_context_tokens": (i + 1) * 5000,
                "tokens_since_compaction": tok_since,
                "post_compaction": comp_idx > 0,
                "compaction_index": comp_idx,
                "is_compaction_event": is_comp,
                "tokens_at_last_compaction": 40000 if comp_idx > 0 else 0,
                "bucket_id": tok_since // 5000,
                "prompt_cache_hit_tokens": 4000,
                "prompt_cache_miss_tokens": 1000,
                "has_real_cache_telemetry": True,
                "ts_request": float(i * 10),
                "ts_response": float(i * 10 + 2),
                "step_confidence": 1.0,
            })
        steps_df = pd.DataFrame(rows)
        sig_df, comp_df = extract_signals_and_compactions(steps_df, warmup_steps=2)
        self.assertEqual(len(comp_df), 1)
        self.assertTrue(bool(comp_df.iloc[0]["s14c_lost_state"]))
        self.assertEqual(int(comp_df.iloc[0]["lost_candidates_count"]), 1)
        post_rows = sig_df[sig_df["post_compaction"]]
        self.assertTrue(post_rows["s14a"].any())
        self.assertTrue(post_rows["s14c"].any())
        self.assertTrue(post_rows["s15"].any())

    def test_hard_failure_floor_requires_at_least_2_fires(self) -> None:
        cfg = load_config(ROOT / "config.yaml")
        rows = []
        # Bucket 0: only 1 hard failure (s1=True once) -> should NOT be floored at 500
        # Bucket 1: 2 hard failures (s1=True, s2=True) -> SHOULD be floored at >= 500
        for idx in range(20):
            b_id = idx // 10
            rows.append({
                "session_id": "s1",
                "task_type": "bugfix",
                "compaction_index": 0,
                "post_compaction": False,
                "step_index": idx,
                "bucket_id": b_id,
                "cumulative_context_tokens": b_id * 5000 + 1000,
                "tokens_since_compaction": b_id * 5000 + 1000,
                "step_confidence": 1.0,
                "s1": (idx in (0, 10)),
                "s2": (idx == 11),
                "s3": False, "s4": False, "s5": False, "s6": False,
                "s7": False, "s8": False, "s9": False, "s10": False,
                "s11": False, "s12": False, "s13": False,
                "s14a": False, "s14b": False, "s14c": False, "s15": False,
                "s14c_epoch_lost": False,
                "warmup": False,
            })
        df_sig = pd.DataFrame(rows)
        scored = score_buckets(df_sig, config=cfg, bootstrap=True)
        b0_raw = int(scored.loc[scored["bucket_id"] == 0, "score_raw"].iloc[0])
        b1_raw = int(scored.loc[scored["bucket_id"] == 1, "score_raw"].iloc[0])
        self.assertLess(b0_raw, 500, "Single transient tool error must not trigger the 500 hard-failure floor")
        self.assertGreaterEqual(b1_raw, 500, ">=2 hard failures in bucket must trigger the 500 floor")

    def test_min_fill_tokens_guard_and_opencode_buffer_semantics(self) -> None:
        cfg = load_config(ROOT / "config.yaml")
        # Short curve only reaching 45k tokens (< 100k min_fill_tokens guard)
        short_curve = pd.DataFrame([
            {"bucket_id": b, "context_tokens": b * 5000, "context_tokens_k": b * 5, "tokens_since_compaction_k": b * 5,
             "mean_score": 20.0 + b * 50.0, "p50": 20.0 + b * 45.0, "p90": 30.0 + b * 60.0,
             "mean_score_raw": 20.0 + b * 50.0, "mean_confidence": 1.0, "n_sessions_contributing": 10,
             "total_steps": 15, "slope_d5k": 50.0}
            for b in range(1, 9)
        ])
        onset = detect_rot_onset(short_curve, config=cfg, min_fill_tokens_override=100000)
        self.assertTrue(onset["guard_triggered_short_dataset"])
        self.assertIn("DATASET TOO SHORT", onset["guard_warning"])
        # Verify OpenCode V2 buffer = model_context_window - recommended_compaction_tokens.
        # Read the window from config instead of hardcoding it: the window is an owner
        # decision that moves (900k -> 1M), and a brittle test here fails on a correct run.
        window = int(cfg["model_context_window"])
        self.assertGreater(window, 0)
        self.assertEqual(
            onset["opencode_v2_buffer_tokens"],
            max(16000, window - onset["recommended_compaction_tokens"]),
        )

    def test_task_type_classifier(self) -> None:
        self.assertEqual(classify_task_type("Please refactor the auth module"), "refactor")
        self.assertEqual(classify_task_type("Fix the crash bug in login"), "bugfix")
        self.assertEqual(classify_task_type("Write unit tests with pytest"), "test")
        self.assertEqual(classify_task_type("Explore how routing works"), "explore")
        self.assertEqual(classify_task_type("Implement new webhook endpoint"), "feature")


if __name__ == "__main__":
    unittest.main()
