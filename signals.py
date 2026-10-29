#!/usr/bin/env python3
"""
Phase 2 — Signal Extraction (signals.py) [V2: 15 Signals / 17 Evaluators + compaction_events]

Extracts all 15 Context Rot signals (s1..s13, s14a, s14b, s14c, s15) per assistant step
and builds the `compaction_events` table in DuckDB (`data/steps.duckdb`) and
`outputs/compaction_events.csv`.

Signals:
  s1:   Tool error (Stateless, weight 1.3)
  s2:   Edit failure (Stateless, weight 1.3)
  s3:   Retry loop within 3 steps after error (Windowed, weight 1.3)
  s4:   Session re-read (Session-stateful, weight 1.3)
  s5:   Self-correction regex (Stateless, weight 0.8)
  s6:   Duplicate read within 5 steps (Windowed, weight 0.8)
  s7:   Unreferenced large output >5k tokens in next 3 steps (Lookahead, weight 0.6)
  s8:   Repetition entropy drop over last 20 tools (Windowed, weight 0.6)
  s9:   Instruction amnesia vs user negative constraints (Session-stateful, weight 0.3)
  s10:  Stale todo snapshot >=10 steps ago, absent in last 5 (Session-stateful, weight 0.3)
  s11:  Contradictory edit — revert to 2 edits ago OR >=3 edits in 20 steps (Session-stateful, weight 0.6)
  s12:  Latency outlier > 3x session median (Session-stateful, weight 0.3)
  s13:  AGENTS.md retrieval failure — violates parsed AGENTS.md rules (Stateless/Rule-based, weight 1.5)
  s14a: Memory write latency — post-compaction steps to first memory write > max_latency (Compaction-window, weight 0.5)
  s14b: Memory write burst/starvation — count in first 10 post-compaction steps == 0 (when facts exist) or >= 5 (Compaction-window, weight 0.5)
  s14c: Lost memory write — candidate facts ("note that X", "remember X", "important: X") from pre-compaction
        missing in post-compaction memory writes (Cross-compaction lookahead, weight 1.2)
  s15:  Compaction recovery failure — post_compaction_rot_score / pre_compaction_rot_score > threshold (Compaction-epoch, weight 1.3)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Set, Tuple

import numpy as np
import pandas as pd
import yaml

from ingest import (
    count_tokens,
    load_agents_rules_from_duckdb,
    load_steps_from_duckdb,
    open_duckdb,
    parse_agents_md_rules,
)


ALL_SIGNAL_COLUMNS = [
    "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12",
    "s13", "s14a", "s14b", "s14c", "s15",
]

EDIT_TOOLS = {"edit", "write", "patch", "multiedit", "str_replace_editor", "apply_patch"}
READ_TOOLS = {"read", "read_file", "view_file", "cat"}

EDIT_FAILURE_REGEX = re.compile(
    r"(string not found|no match|old_string not found|failed to apply patch|"
    r"hunk #\d+ failed|conflict|could not find|replacement failed|0 occurrences|"
    r"exact match not found|search block not found)",
    re.IGNORECASE,
)

SELF_CORRECTION_REGEX = re.compile(
    r"\b(I apologi[sz]e|let me fix|my mistake|sorry about that|"
    r"I made an error|wait,\s*that'?s wrong|oops|let me correct|"
    r"I misread|my bad|apologies for the confusion)\b",
    re.IGNORECASE,
)

NEGATIVE_CONSTRAINT_REGEX = re.compile(
    r"(?:don'?t|do\s+not|never|must\s+not|avoid|stop)\s+"
    r"(?:use|using|call|calling|modify|modifying|edit|editing|touch|touching|"
    r"write\s+to|run|running|import|importing|delete|deleting|change|changing)\s+"
    r"[`'\"]?([A-Za-z0-9_./\-]+)[`'\"]?",
    re.IGNORECASE,
)

# Signal 14c: Candidate memory facts in pre-compaction assistant text
CANDIDATE_FACT_REGEX = re.compile(
    r"(?:note\s+that|remember(?:\s+that)?|important:|invariant:|key\s+decision:)\s*"
    r"([A-Za-z0-9_/:=\-]+)",
    re.IGNORECASE,
)


def canonical_json(obj: Any) -> str:
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return str(obj)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def extract_path_from_params(params: Dict[str, Any]) -> Optional[str]:
    if not isinstance(params, dict):
        return None
    for key in ("path", "file_path", "filepath", "target_file", "file", "filename"):
        val = params.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def compute_shannon_entropy(seq: List[str]) -> float:
    if not seq:
        return 0.0
    n = len(seq)
    counts = Counter(seq)
    entropy = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def extract_80char_shingles(text: str, max_shingles: int = 250) -> List[str]:
    if not text:
        return []
    shingles: List[str] = []
    if len(text) >= 80:
        stride = max(20, (len(text) - 80) // max(1, (max_shingles // 2)))
        for i in range(0, len(text) - 80 + 1, stride):
            shingles.append(text[i : i + 80])
            if len(shingles) >= max_shingles // 2:
                break

    norm = re.sub(r"\s+", " ", text).strip()
    if len(norm) >= 80:
        stride = max(20, (len(norm) - 80) // max(1, (max_shingles // 2)))
        for i in range(0, len(norm) - 80 + 1, stride):
            shingles.append(norm[i : i + 80])
            if len(shingles) >= max_shingles:
                break

    return shingles


def extract_candidate_facts(text: str) -> List[str]:
    """Extracts normalized candidate fact keys from assistant text for Signal 14c."""
    if not text:
        return []
    facts: List[str] = []
    for m in CANDIDATE_FACT_REGEX.finditer(text):
        phrase = m.group(1).strip().rstrip(".,;:")
        if len(phrase) >= 5:
            facts.append(phrase.lower())
    return facts


# ---------------------------------------------------------------------------
# Session State for Stateful & Windowed Reducers
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    assistant_step_pos: int = 0
    failed_tool_hashes_history: List[Tuple[int, Set[str]]] = field(default_factory=list)
    paths_read_steps: Dict[str, List[int]] = field(default_factory=dict)
    tool_seq: List[str] = field(default_factory=list)
    early_entropies: List[float] = field(default_factory=list)
    negative_constraints: Set[str] = field(default_factory=set)
    todo_hash_steps: Dict[str, List[int]] = field(default_factory=dict)
    file_edit_hashes: Dict[str, List[str]] = field(default_factory=dict)
    file_edit_steps: Dict[str, List[int]] = field(default_factory=dict)
    session_median_latency: float = 1.0


# ---------------------------------------------------------------------------
# Signal Evaluators (s1..s13)
# ---------------------------------------------------------------------------

def eval_s1_tool_error(tool_calls: List[Dict[str, Any]]) -> bool:
    for tc in tool_calls:
        err = tc.get("error")
        if err is not None and str(err).strip() != "":
            return True
        exit_code = tc.get("exit_code")
        if exit_code is not None and int(exit_code) != 0:
            return True
    return False


def eval_s2_edit_failure(tool_calls: List[Dict[str, Any]]) -> bool:
    for tc in tool_calls:
        name = str(tc.get("name", "")).lower()
        if name in EDIT_TOOLS:
            combined_text = f"{tc.get('result', '') or ''} {tc.get('error', '') or ''}"
            if EDIT_FAILURE_REGEX.search(combined_text):
                return True
    return False


def eval_s3_retry_loop(tool_calls: List[Dict[str, Any]], state: SessionState) -> bool:
    if not tool_calls:
        return False
    cur_pos = state.assistant_step_pos
    recent_failed_hashes: Set[str] = set()
    for past_pos, failed_set in state.failed_tool_hashes_history:
        if 1 <= (cur_pos - past_pos) <= 3:
            recent_failed_hashes.update(failed_set)
    if not recent_failed_hashes:
        return False
    for tc in tool_calls:
        name = str(tc.get("name", "")).lower()
        params = tc.get("params", {})
        h = sha256_hex(name + canonical_json(params))
        if h in recent_failed_hashes:
            return True
    return False


def eval_s4_reread(tool_calls: List[Dict[str, Any]], state: SessionState) -> bool:
    for tc in tool_calls:
        name = str(tc.get("name", "")).lower()
        if name in READ_TOOLS:
            path = extract_path_from_params(tc.get("params", {}))
            if path and path in state.paths_read_steps and len(state.paths_read_steps[path]) > 0:
                return True
    return False


def eval_s5_self_correction(text: str) -> bool:
    if not text:
        return False
    return bool(SELF_CORRECTION_REGEX.search(text))


def eval_s6_duplicate_read(tool_calls: List[Dict[str, Any]], state: SessionState) -> bool:
    cur_pos = state.assistant_step_pos
    for tc in tool_calls:
        name = str(tc.get("name", "")).lower()
        if name in READ_TOOLS:
            path = extract_path_from_params(tc.get("params", {}))
            if path and path in state.paths_read_steps:
                for prev_pos in state.paths_read_steps[path]:
                    if 1 <= (cur_pos - prev_pos) <= 5:
                        return True
    return False


def eval_s7_unreferenced_large_output(
    tool_calls: List[Dict[str, Any]],
    subsequent_assistant_texts: List[str],
) -> bool:
    for tc in tool_calls:
        res = tc.get("result")
        if res is None:
            continue
        res_str = res if isinstance(res, str) else canonical_json(res)
        res_tokens = tc.get("result_tokens")
        if res_tokens is None or int(res_tokens) == 0:
            res_tokens = count_tokens(res_str)
        if int(res_tokens) <= 5000:
            continue

        shingles = extract_80char_shingles(res_str)
        if not shingles:
            continue

        referenced = False
        for next_txt in subsequent_assistant_texts[:3]:
            if not next_txt:
                continue
            norm_next = re.sub(r"\s+", " ", next_txt)
            for sh in shingles:
                if sh in next_txt or sh in norm_next:
                    referenced = True
                    break
            if referenced:
                break

        if not referenced:
            return True
    return False


def eval_s8_repetition_entropy_drop(tool_calls: List[Dict[str, Any]], state: SessionState) -> bool:
    if not tool_calls:
        return False
    candidate_seq = state.tool_seq + [str(tc.get("name", "")).lower() for tc in tool_calls]
    if len(candidate_seq) < 20:
        return False
    window_20 = candidate_seq[-20:]
    h_current = compute_shannon_entropy(window_20)
    h_baseline = float(np.median(state.early_entropies)) if state.early_entropies else 1.5
    return h_current < (h_baseline - 0.3)


def eval_s9_instruction_amnesia(
    text: str,
    tool_calls: List[Dict[str, Any]],
    state: SessionState,
) -> bool:
    if not state.negative_constraints:
        return False
    for tc in tool_calls:
        t_name = str(tc.get("name", "")).lower()
        params_str = canonical_json(tc.get("params", {})).lower()
        for forbidden in state.negative_constraints:
            if len(forbidden) < 3:
                continue
            if forbidden == t_name or forbidden in params_str:
                return True
    text_lower = (text or "").lower()
    for forbidden in state.negative_constraints:
        if len(forbidden) >= 4 and (
            f"import {forbidden}" in text_lower
            or f"from {forbidden}" in text_lower
            or f"editing {forbidden}" in text_lower
        ):
            return True
    return False


def eval_s10_stale_todo_snapshot(todos: Optional[List[Dict[str, Any]]], state: SessionState) -> bool:
    if not todos:
        return False
    cur_pos = state.assistant_step_pos
    h = sha256_hex(canonical_json(todos))
    past_positions = state.todo_hash_steps.get(h, [])
    if not past_positions:
        return False
    seen_ge_10_ago = any((cur_pos - p) >= 10 for p in past_positions)
    seen_in_last_5 = any(1 <= (cur_pos - p) <= 5 for p in past_positions)
    return seen_ge_10_ago and (not seen_in_last_5)


def eval_s11_contradictory_edit(tool_calls: List[Dict[str, Any]], state: SessionState) -> bool:
    cur_pos = state.assistant_step_pos
    for tc in tool_calls:
        name = str(tc.get("name", "")).lower()
        if name not in EDIT_TOOLS:
            continue
        params = tc.get("params", {})
        path = extract_path_from_params(params)
        if not path:
            continue
        new_state_material = (
            params.get("new_string")
            or params.get("content")
            or params.get("patch")
            or params.get("diff")
            or tc.get("result")
            or canonical_json(params)
        )
        new_hash = sha256_hex(str(new_state_material))
        prior_hashes = state.file_edit_hashes.get(path, [])
        prior_steps = state.file_edit_steps.get(path, [])

        if len(prior_hashes) >= 2 and new_hash == prior_hashes[-2]:
            return True
        recent_edits_in_20 = [p for p in prior_steps if (cur_pos - p) < 20]
        if len(recent_edits_in_20) + 1 >= 3:
            return True
    return False


def eval_s12_latency_outlier(ts_request: float, ts_response: float, state: SessionState) -> bool:
    latency = max(0.0, float(ts_response) - float(ts_request))
    if state.session_median_latency <= 0:
        return False
    return latency > (3.0 * state.session_median_latency)


def eval_s13_agents_md_failure(
    tool_calls: List[Dict[str, Any]],
    compiled_rules: List[Pattern[str]],
) -> bool:
    """
    Signal 13 (weight 1.5): Detects violations of discrete AGENTS.md rules
    in tool-call sequences (paths, commands, code edits).
    """
    if not tool_calls or not compiled_rules:
        return False
    for tc in tool_calls:
        params_blob = canonical_json(tc.get("params", {}))
        name = str(tc.get("name", ""))
        combined = f"{name} {params_blob}"
        for pat in compiled_rules:
            if pat.search(combined):
                return True
    return False


# ---------------------------------------------------------------------------
# State Reducer Update
# ---------------------------------------------------------------------------

def update_session_state(
    state: SessionState,
    role: str,
    text: str,
    tool_calls: List[Dict[str, Any]],
    todos: Optional[List[Dict[str, Any]]],
) -> None:
    if role == "user":
        if text:
            for match in NEGATIVE_CONSTRAINT_REGEX.finditer(text):
                target = match.group(1).strip().lower()
                if len(target) >= 3:
                    state.negative_constraints.add(target)
        return

    cur_pos = state.assistant_step_pos

    failed_hashes: Set[str] = set()
    for tc in tool_calls:
        name = str(tc.get("name", "")).lower()
        params = tc.get("params", {})
        err = tc.get("error")
        exit_code = tc.get("exit_code")
        is_err = (err is not None and str(err).strip() != "") or (exit_code is not None and int(exit_code) != 0)
        if is_err:
            failed_hashes.add(sha256_hex(name + canonical_json(params)))
    if failed_hashes:
        state.failed_tool_hashes_history.append((cur_pos, failed_hashes))

    for tc in tool_calls:
        name = str(tc.get("name", "")).lower()
        if name in READ_TOOLS:
            path = extract_path_from_params(tc.get("params", {}))
            if path:
                state.paths_read_steps.setdefault(path, []).append(cur_pos)

    for tc in tool_calls:
        name = str(tc.get("name", "")).lower()
        state.tool_seq.append(name)
        if cur_pos < 50 and len(state.tool_seq) >= 20:
            state.early_entropies.append(compute_shannon_entropy(state.tool_seq[-20:]))

    if todos:
        h = sha256_hex(canonical_json(todos))
        state.todo_hash_steps.setdefault(h, []).append(cur_pos)

    for tc in tool_calls:
        name = str(tc.get("name", "")).lower()
        if name in EDIT_TOOLS:
            params = tc.get("params", {})
            path = extract_path_from_params(params)
            if path:
                new_state_material = (
                    params.get("new_string")
                    or params.get("content")
                    or params.get("patch")
                    or params.get("diff")
                    or tc.get("result")
                    or canonical_json(params)
                )
                new_hash = sha256_hex(str(new_state_material))
                state.file_edit_hashes.setdefault(path, []).append(new_hash)
                state.file_edit_steps.setdefault(path, []).append(cur_pos)

    state.assistant_step_pos += 1


# ---------------------------------------------------------------------------
# Instantaneous Rot Proxy for Compaction Recovery Ratio (Signal 15)
# ---------------------------------------------------------------------------

def _compute_window_rot_score(window_rows: List[Dict[str, Any]]) -> float:
    """
    Computes a local 1-1000 rot score across a window of assistant steps (e.g. 10 steps
    pre-compaction vs 10 steps post-compaction) using Signals 1..14c.
    """
    if not window_rows:
        return 1.0
    weights = {
        "s1": 1.3, "s2": 1.3, "s3": 1.3, "s4": 1.3,
        "s5": 0.8, "s6": 0.8, "s7": 0.6, "s8": 0.6,
        "s9": 0.3, "s10": 0.3, "s11": 0.6, "s12": 0.3,
        "s13": 1.5, "s14a": 0.5, "s14b": 0.5, "s14c": 1.2,
    }
    priors = {
        "s1": 0.14, "s2": 0.09, "s3": 0.07, "s4": 0.16,
        "s5": 0.11, "s6": 0.10, "s7": 0.06, "s8": 0.08,
        "s9": 0.04, "s10": 0.04, "s11": 0.07, "s12": 0.05,
        "s13": 0.08, "s14a": 0.07, "s14b": 0.06, "s14c": 0.09,
    }
    n = len(window_rows)
    w_sum = 0.0
    total_w = sum(weights.values())
    hard_fires = 0
    for sig, w in weights.items():
        fires = sum(1 for r in window_rows if r.get(sig, False))
        if sig in ("s1", "s2"):
            hard_fires += fires
        rate = fires / n
        norm = min(1.0, rate / priors[sig])
        w_sum += w * norm
    score = round(1.0 + (w_sum / total_w) * 999.0, 1)
    if hard_fires >= 2:
        score = max(score, 500.0)
    return float(score)


# ---------------------------------------------------------------------------
# Full Extraction Pipeline (Signals 1..15 + Compaction Events Table)
# ---------------------------------------------------------------------------

def extract_signals_and_compactions(
    steps_df: pd.DataFrame,
    agents_rules: Optional[List[Dict[str, str]]] = None,
    warmup_steps: int = 5,
    memory_write_max_latency_steps: int = 3,
    memory_write_burst_window_steps: int = 10,
    memory_write_burst_threshold: int = 5,
    recovery_ratio_threshold: float = 0.65,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs the 15-signal extraction pipeline across all sessions and builds both:
      1. `step_signals` DataFrame (per assistant step)
      2. `compaction_events` DataFrame (one row per compaction event)
    """
    if steps_df.empty:
        sig_cols = [
            "session_id", "task_type", "step_index", "bucket_id",
            "cumulative_context_tokens", "tokens_since_compaction",
            "post_compaction", "compaction_index", "is_compaction_event",
            "prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "has_real_cache_telemetry",
            "tokens_output", "step_confidence",
            *ALL_SIGNAL_COLUMNS, "s14c_epoch_lost", "warmup",
        ]
        comp_cols = [
            "session_id", "task_type", "compaction_index", "step_index",
            "tokens_at_compaction", "pre_rot_score", "post_rot_score",
            "recovery_ratio", "memory_writes_after", "memory_write_latency_steps",
            "candidate_facts_count", "lost_candidates_count", "s14c_lost_state", "s15_failed_recovery",
        ]
        return pd.DataFrame(columns=sig_cols), pd.DataFrame(columns=comp_cols)

    rules_list = agents_rules or parse_agents_md_rules()
    compiled_agents_rules = [re.compile(r["pattern"], re.IGNORECASE) for r in rules_list if r.get("pattern")]

    emitted_rows: List[Dict[str, Any]] = []
    compaction_event_rows: List[Dict[str, Any]] = []

    for session_id, group in steps_df.groupby("session_id", sort=False):
        rows = group.sort_values("step_index").to_dict("records")
        task_type = str(rows[0].get("task_type", "feature")) if rows else "feature"

        assistant_indices: List[int] = []
        latencies: List[float] = []

        for idx, r in enumerate(rows):
            tc_raw = r.get("tool_calls")
            r["_tool_calls"] = json.loads(tc_raw) if isinstance(tc_raw, str) and tc_raw else (tc_raw if isinstance(tc_raw, list) else [])
            td_raw = r.get("todos")
            r["_todos"] = json.loads(td_raw) if isinstance(td_raw, str) and td_raw else (td_raw if isinstance(td_raw, list) else None)
            mw_raw = r.get("memory_writes")
            r["_memory_writes"] = json.loads(mw_raw) if isinstance(mw_raw, str) and mw_raw else (mw_raw if isinstance(mw_raw, list) else [])
            r["_text"] = str(r.get("text") or "").strip()

            if r.get("role") == "assistant":
                if not r["_text"] and not r["_tool_calls"] and not r["_todos"] and not r["_memory_writes"]:
                    r["_skip_empty"] = True
                else:
                    r["_skip_empty"] = False
                    assistant_indices.append(idx)
                    lat = max(0.0, float(r.get("ts_response", 0.0)) - float(r.get("ts_request", 0.0)))
                    if lat > 0:
                        latencies.append(lat)

        state = SessionState(
            session_median_latency=float(np.median(latencies)) if latencies else 1.0
        )
        pos_in_assistant = {row_idx: pos for pos, row_idx in enumerate(assistant_indices)}

        # Pass 1: Evaluate s1..s13 for every assistant step in the session
        sess_emitted: List[Dict[str, Any]] = []
        for idx, r in enumerate(rows):
            role = r.get("role", "assistant")
            text = r["_text"]
            tool_calls = r["_tool_calls"]
            todos = r["_todos"]

            if role == "user":
                update_session_state(state, role, text, tool_calls, todos)
                continue

            if r.get("_skip_empty", False):
                continue

            a_pos = pos_in_assistant[idx]
            next_assistant_texts = [
                rows[assistant_indices[next_pos]]["_text"]
                for next_pos in range(a_pos + 1, min(len(assistant_indices), a_pos + 4))
            ]

            is_tool_only = (text == "" and len(tool_calls) > 0)

            s1 = eval_s1_tool_error(tool_calls)
            s2 = eval_s2_edit_failure(tool_calls)
            s3 = eval_s3_retry_loop(tool_calls, state)
            s4 = eval_s4_reread(tool_calls, state)
            s5 = False if is_tool_only else eval_s5_self_correction(text)
            s6 = eval_s6_duplicate_read(tool_calls, state)
            s7 = eval_s7_unreferenced_large_output(tool_calls, next_assistant_texts)
            s8 = eval_s8_repetition_entropy_drop(tool_calls, state)
            s9 = eval_s9_instruction_amnesia(text, tool_calls, state)
            s10 = eval_s10_stale_todo_snapshot(todos, state)
            s11 = eval_s11_contradictory_edit(tool_calls, state)
            s12 = eval_s12_latency_outlier(
                float(r.get("ts_request", 0.0)),
                float(r.get("ts_response", 0.0)),
                state,
            )
            s13 = eval_s13_agents_md_failure(tool_calls, compiled_agents_rules)

            is_warmup = state.assistant_step_pos < warmup_steps

            sess_emitted.append({
                "session_id": str(session_id),
                "task_type": task_type,
                "step_index": int(r["step_index"]),
                "bucket_id": int(r["bucket_id"]),
                "cumulative_context_tokens": int(r["cumulative_context_tokens"]),
                "tokens_since_compaction": int(r.get("tokens_since_compaction", r["cumulative_context_tokens"])),
                "post_compaction": bool(r.get("post_compaction", False)),
                "compaction_index": int(r.get("compaction_index", 0)),
                "is_compaction_event": bool(r.get("is_compaction_event", False)),
                "tokens_at_last_compaction": int(r.get("tokens_at_last_compaction", 0)),
                "prompt_cache_hit_tokens": int(r.get("prompt_cache_hit_tokens", 0)),
                "prompt_cache_miss_tokens": int(r.get("prompt_cache_miss_tokens", 0)),
                "has_real_cache_telemetry": bool(r.get("has_real_cache_telemetry", False)),
                "tokens_output": int(r.get("tokens_output", 0)),
                "step_confidence": float(r.get("step_confidence", 0.5 if is_warmup else 1.0)),
                "s1": bool(s1),
                "s2": bool(s2),
                "s3": bool(s3),
                "s4": bool(s4),
                "s5": bool(s5),
                "s6": bool(s6),
                "s7": bool(s7),
                "s8": bool(s8),
                "s9": bool(s9),
                "s10": bool(s10),
                "s11": bool(s11),
                "s12": bool(s12),
                "s13": bool(s13),
                "s14a": False,
                "s14b": False,
                "s14c": False,
                "s15": False,
                "s14c_epoch_lost": False,
                "warmup": bool(is_warmup),
                "_text": text,
                "_memory_writes": r["_memory_writes"],
            })

            update_session_state(state, role, text, tool_calls, todos)

        # Pass 2: Evaluate Compaction Events & Signals 14a, 14b, 14c, and 15
        compaction_positions = [
            pos for pos, sr in enumerate(sess_emitted) if sr["is_compaction_event"]
        ]

        for c_idx_pos, comp_pos in enumerate(compaction_positions):
            prev_comp_pos = compaction_positions[c_idx_pos - 1] if c_idx_pos > 0 else 0
            next_comp_pos = compaction_positions[c_idx_pos + 1] if (c_idx_pos + 1) < len(compaction_positions) else len(sess_emitted)

            pre_epoch_rows = sess_emitted[prev_comp_pos:comp_pos]
            post_epoch_rows = sess_emitted[comp_pos:next_comp_pos]
            post_window_10 = post_epoch_rows[:memory_write_burst_window_steps]

            # 1. Extract candidate facts from pre-compaction assistant text (Signal 14c)
            candidate_facts: List[str] = []
            for pr in pre_epoch_rows:
                candidate_facts.extend(extract_candidate_facts(pr["_text"]))
            unique_candidates = sorted(list(set(candidate_facts)))

            # Collect memory writes in the last 3 pre-compaction steps (gate closing) + first 10 post-compaction steps
            gate_memory_blobs: List[str] = []
            for pr in pre_epoch_rows[-3:]:
                gate_memory_blobs.extend(pr["_memory_writes"])
            for por in post_window_10:
                gate_memory_blobs.extend(por["_memory_writes"])
                gate_memory_blobs.append(por["_text"])
            combined_memory_text = " ".join(gate_memory_blobs).lower()

            lost_candidates = [
                fact for fact in unique_candidates if fact not in combined_memory_text
            ]
            lost_count = len(lost_candidates)
            s14c_fired = bool(lost_count > 0)

            # 2. Signal 14a (Memory write latency steps after compaction) & 14b (Memory write burst count in first 10 steps)
            latency_steps: int = memory_write_burst_window_steps + 1
            memory_writes_after_count = 0
            for offset, por in enumerate(post_window_10):
                mw_list = por["_memory_writes"]
                if mw_list:
                    if latency_steps > memory_write_burst_window_steps:
                        latency_steps = offset
                    memory_writes_after_count += len(mw_list)

            s14a_fired = bool(latency_steps > memory_write_max_latency_steps)
            s14b_fired = bool(
                (memory_writes_after_count == 0 and len(unique_candidates) > 0)
                or (memory_writes_after_count >= memory_write_burst_threshold)
            )

            # Mark s14a, s14b, s14c on the post-compaction steps
            for offset, por in enumerate(post_epoch_rows):
                por["s14c_epoch_lost"] = s14c_fired
                if offset < memory_write_burst_window_steps:
                    por["s14a"] = s14a_fired
                    por["s14b"] = s14b_fired
                    por["s14c"] = s14c_fired

            # 3. Signal 15: Compaction recovery ratio = post_compaction_rot_score / pre_compaction_rot_score
            pre_10 = pre_epoch_rows[-10:] if len(pre_epoch_rows) >= 1 else []
            pre_rot_score = _compute_window_rot_score(pre_10)
            post_rot_score = _compute_window_rot_score(post_window_10)
            recovery_ratio = round(post_rot_score / max(1.0, pre_rot_score), 4)
            s15_fired = bool(recovery_ratio > recovery_ratio_threshold and post_rot_score >= 180.0)

            for por in post_epoch_rows:
                por["s15"] = s15_fired

            comp_step_row = sess_emitted[comp_pos]
            tokens_at_comp = int(comp_step_row.get("tokens_at_last_compaction") or 0)
            if tokens_at_comp <= 0 and pre_epoch_rows:
                tokens_at_comp = int(pre_epoch_rows[-1]["tokens_since_compaction"])

            compaction_event_rows.append({
                "session_id": str(session_id),
                "task_type": task_type,
                "compaction_index": int(comp_step_row["compaction_index"]),
                "step_index": int(comp_step_row["step_index"]),
                "tokens_at_compaction": tokens_at_comp,
                "pre_rot_score": round(pre_rot_score, 1),
                "post_rot_score": round(post_rot_score, 1),
                "recovery_ratio": recovery_ratio,
                "memory_writes_after": int(memory_writes_after_count),
                "memory_write_latency_steps": int(latency_steps),
                "candidate_facts_count": len(unique_candidates),
                "lost_candidates_count": int(lost_count),
                "s14c_lost_state": s14c_fired,
                "s15_failed_recovery": s15_fired,
            })

        # Strip temporary private fields before appending
        for sr in sess_emitted:
            sr.pop("_text", None)
            sr.pop("_memory_writes", None)
            sr.pop("tokens_at_last_compaction", None)
            emitted_rows.append(sr)

    df_signals = pd.DataFrame(emitted_rows)
    df_compactions = pd.DataFrame(compaction_event_rows)
    return df_signals, df_compactions


def extract_signals_from_steps(
    steps_df: pd.DataFrame,
    warmup_steps: int = 5,
    agents_rules: Optional[List[Dict[str, str]]] = None,
) -> pd.DataFrame:
    """Backward-compatible wrapper returning `step_signals` DataFrame."""
    df_sig, _ = extract_signals_and_compactions(
        steps_df,
        agents_rules=agents_rules,
        warmup_steps=warmup_steps,
    )
    return df_sig


def save_signals_to_duckdb(
    df_signals: pd.DataFrame,
    df_compactions: pd.DataFrame,
    duckdb_path: str | Path,
) -> None:
    duckdb_path = Path(duckdb_path).expanduser().resolve()
    with open_duckdb(duckdb_path, read_only=False) as con:
        con.execute("DROP TABLE IF EXISTS step_signals")
        con.register("df_sig_view", df_signals)
        con.execute("CREATE TABLE step_signals AS SELECT * FROM df_sig_view")

        con.execute("DROP TABLE IF EXISTS compaction_events")
        if not df_compactions.empty:
            con.register("df_comp_view", df_compactions)
            con.execute("CREATE TABLE compaction_events AS SELECT * FROM df_comp_view")
        else:
            con.execute("""
                CREATE TABLE compaction_events (
                    session_id TEXT, task_type TEXT, compaction_index BIGINT, step_index BIGINT,
                    tokens_at_compaction BIGINT, pre_rot_score DOUBLE, post_rot_score DOUBLE,
                    recovery_ratio DOUBLE, memory_writes_after BIGINT, memory_write_latency_steps BIGINT,
                    candidate_facts_count BIGINT, lost_candidates_count BIGINT,
                    s14c_lost_state BOOLEAN, s15_failed_recovery BOOLEAN
                )
            """)


def run_signal_extraction(
    duckdb_path: str | Path = "data/steps.duckdb",
    config_path: Optional[str | Path] = "config.yaml",
    out_csv: Optional[str | Path] = "outputs/step_signals.csv",
    compactions_csv: Optional[str | Path] = "outputs/compaction_events.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    warmup_steps = 5
    max_lat = 3
    burst_win = 10
    burst_thresh = 5
    rec_thresh = 0.65

    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            warmup_steps = int(cfg.get("warmup_steps", 5))
            max_lat = int(cfg.get("memory_write_max_latency_steps", 3))
            burst_win = int(cfg.get("memory_write_burst_window_steps", 10))
            burst_thresh = int(cfg.get("memory_write_burst_threshold", 5))
            rec_thresh = float(cfg.get("recovery_ratio_threshold", 0.65))

    steps_df = load_steps_from_duckdb(duckdb_path)
    agents_rules = load_agents_rules_from_duckdb(duckdb_path)

    df_signals, df_compactions = extract_signals_and_compactions(
        steps_df,
        agents_rules=agents_rules,
        warmup_steps=warmup_steps,
        memory_write_max_latency_steps=max_lat,
        memory_write_burst_window_steps=burst_win,
        memory_write_burst_threshold=burst_thresh,
        recovery_ratio_threshold=rec_thresh,
    )
    save_signals_to_duckdb(df_signals, df_compactions, duckdb_path)

    if out_csv:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_signals.to_csv(out_path, index=False)

    if compactions_csv:
        comp_path = Path(compactions_csv)
        comp_path.parent.mkdir(parents=True, exist_ok=True)
        df_compactions.to_csv(comp_path, index=False)

    return df_signals, df_compactions


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: Extract 15 context rot signals and compaction_events.")
    parser.add_argument("--duckdb", type=str, default="data/steps.duckdb", help="Path to steps.duckdb")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--out-csv", type=str, default="outputs/step_signals.csv", help="Output CSV for step signals")
    parser.add_argument("--out-compactions", type=str, default="outputs/compaction_events.csv", help="Output CSV for compaction events")
    args = parser.parse_args()

    df_sig, df_comp = run_signal_extraction(args.duckdb, args.config, args.out_csv, args.out_compactions)
    non_warmup = df_sig[~df_sig["warmup"].astype(bool)] if not df_sig.empty else df_sig
    print(f"[Phase 2: Signals] Evaluated {len(df_sig)} assistant steps ({len(non_warmup)} non-warmup) & {len(df_comp)} compaction events.")
    if not non_warmup.empty:
        fire_counts = {col: int(non_warmup[col].astype(bool).sum()) for col in ALL_SIGNAL_COLUMNS}
        print(f"[Phase 2: Signals] Non-warmup fire counts: {fire_counts}")


if __name__ == "__main__":
    main()
