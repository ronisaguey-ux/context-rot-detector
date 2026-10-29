#!/usr/bin/env python3
"""
Phase 1 — Ingestion Layer (ingest.py) [V2 Architecture-Specific]

Key capabilities:
  1. Reads OpenCode SQLite (`session`, `message`, `part` tables with JSON `part.data`).
  2. Tracks compaction boundaries per session and computes BOTH:
       - `cumulative_context_tokens` (raw cumulative tokens from session start)
       - `tokens_since_compaction` (context fill since most recent compaction or start)
       - `post_compaction` (bool), `compaction_index` (int), `is_compaction_event` (bool),
         `tokens_at_last_compaction` (int)
     Buckets strictly on `bucket_id = floor(tokens_since_compaction / bucket_size_tokens)`.
  3. Captures real prompt-cache usage telemetry (`prompt_cache_hit_tokens`,
     `prompt_cache_miss_tokens`, `has_real_cache_telemetry`) from `part.data` / `message.data`.
  4. Tags every session with a coarse `task_type` (`refactor`, `bugfix`, `feature`, `test`, `explore`)
     extracted from the first user message.
  5. Parses `AGENTS.md` into discrete rule matchers (`agents_rules` table in DuckDB) for Signal 13.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import duckdb
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Filesystem-Safe SQLite & DuckDB Helpers
# ---------------------------------------------------------------------------

def _staging_path(target: Path, suffix: str) -> Path:
    h = hashlib.sha256(str(target.resolve()).encode("utf-8")).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"rot_detector_{h}_{target.name}{suffix}"


@contextmanager
def open_sqlite(db_path: str | Path, write: bool = False) -> Iterator[sqlite3.Connection]:
    target = Path(db_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = _staging_path(target, ".sqlite")
    if target.exists() and target.stat().st_size > 0:
        shutil.copy2(target, staged)
    elif staged.exists() and write:
        staged.unlink()
    conn = sqlite3.connect(str(staged))
    try:
        yield conn
        if write:
            conn.commit()
    finally:
        conn.close()
        if write and staged.exists():
            shutil.copy2(staged, target)


@contextmanager
def open_duckdb(duckdb_path: str | Path, read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    target = Path(duckdb_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = _staging_path(target, ".duckdb")
    staged_wal = Path(str(staged) + ".wal")
    if staged_wal.exists():
        staged_wal.unlink()
    if target.exists() and target.stat().st_size > 0:
        shutil.copy2(target, staged)
    elif staged.exists() and not read_only:
        staged.unlink()
    con = duckdb.connect(str(staged), read_only=read_only)
    if hasattr(con, "_conn"):
        try:
            con._conn.close()
        except Exception:
            pass
        con._conn = sqlite3.connect(str(staged))
    try:
        yield con
        if not read_only:
            if hasattr(con, "_conn"):
                con._conn.commit()
            else:
                con.execute("CHECKPOINT")
    finally:
        con.close()
        if not read_only and staged.exists():
            shutil.copy2(staged, target)


# ---------------------------------------------------------------------------
# Token Accounting
# ---------------------------------------------------------------------------

def _build_token_counter(model_hint: str = "deepseek-ai/DeepSeek-V3") -> Tuple[Callable[[str], int], str]:
    try:
        from transformers import AutoTokenizer  # type: ignore
        tok = AutoTokenizer.from_pretrained(model_hint, local_files_only=True)
        return (lambda text: len(tok.encode(text, add_special_tokens=False)) if text else 0), f"transformers:{model_hint}"
    except Exception:
        pass

    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return (lambda text: len(enc.encode(text, disallowed_special=())) if text else 0), "tiktoken:cl100k_base"
    except Exception:
        pass

    token_pattern = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?\d{1,3}| ?[^\sA-Za-z\d]+|\s+(?!\S)|\s+"""
    )

    def _estimate_bpe_tokens(text: str) -> int:
        if not text:
            return 0
        raw_pieces = token_pattern.findall(text)
        total = 0
        for piece in raw_pieces:
            length = len(piece.strip())
            if length <= 4:
                total += 1
            else:
                total += max(1, math.ceil(length / 4.2))
        return total

    return _estimate_bpe_tokens, "bpe_regex_cl100k_approx"


GLOBAL_TOKEN_COUNTER, TOKENIZER_BACKEND = _build_token_counter()


def count_tokens(text: Any) -> int:
    if text is None:
        return 0
    if not isinstance(text, str):
        try:
            text = json.dumps(text, ensure_ascii=False, sort_keys=True)
        except Exception:
            text = str(text)
    return GLOBAL_TOKEN_COUNTER(text)


# ---------------------------------------------------------------------------
# Task-Type Classifier (Item 12) & AGENTS.md Rule Parser (Item 5)
# ---------------------------------------------------------------------------

TASK_TYPE_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("refactor", re.compile(r"\b(refactor|restructure|clean\s*up|modularize|decouple|rename|migrate)\b", re.I)),
    ("bugfix",   re.compile(r"\b(bug|fix|broken|error|issue|crash|regression|exception|fail|defect)\b", re.I)),
    ("test",     re.compile(r"\b(unit\s*test|integration\s*test|pytest|coverage|test\s*suite|write\s*tests|spec)\b", re.I)),
    ("explore",  re.compile(r"\b(explore|investigate|understand|explain|audit|analyze|trace|where\s+is|how\s+does)\b", re.I)),
    ("feature",  re.compile(r"\b(add|implement|build|create|feature|support|endpoint|introduce|new)\b", re.I)),
]


def classify_task_type(first_user_text: str) -> str:
    """Extracts a coarse task type (refactor / bugfix / feature / test / explore) from the first user message."""
    if not first_user_text:
        return "feature"
    for label, pat in TASK_TYPE_PATTERNS:
        if pat.search(first_user_text):
            return label
    return "feature"


def parse_agents_md_rules(agents_md_path: Optional[str | Path] = "AGENTS.md") -> List[Dict[str, str]]:
    """
    Parses markdown bullets in AGENTS.md into discrete regex rules for Signal 13.
    Extracts forbidden backticked identifiers/commands or negative imperatives.
    """
    default_rules = [
        {"rule_id": "NO_LEGACY_CONFIG",      "pattern": r"legacy_config\.py",             "description": "Never modify or edit legacy_config.py directly"},
        {"rule_id": "NO_RAW_SQL_MIGRATIONS", "pattern": r"(migrations/locked/|DROP\s+TABLE)", "description": "Never edit migrations/locked/ or run DROP TABLE"},
        {"rule_id": "NO_DEPRECATED_CRYPTO",  "pattern": r"(deprecated_crypto|md5_signer)", "description": "Never import or use deprecated_crypto or md5_signer"},
        {"rule_id": "PYTEST_TARGETED",       "pattern": r"pytest\s+--full-suite-unbounded", "description": "Never run pytest --full-suite-unbounded"},
    ]
    if not agents_md_path:
        return default_rules
    p = Path(agents_md_path)
    if not p.exists():
        return default_rules

    parsed_rules: List[Dict[str, str]] = []
    rule_line_re = re.compile(r"^\s*[-*]\s*(?:\[RULE:\s*([A-Za-z0-9_]+)\])?\s*(.+)$")
    backtick_re = re.compile(r"`([^`]+)`")

    for idx, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        m = rule_line_re.match(line)
        if not m:
            continue
        rule_id = m.group(1) or f"AGENTS_RULE_{idx + 1}"
        body = m.group(2).strip()
        # Only turn negative/prohibition rules ("Never...", "Do not...", "Forbidden...") into violation regexes
        if not re.search(r"\b(never|do\s+not|don'?t|forbidden|must\s+not|prohibited)\b", body, re.I):
            continue
        literals = [
            re.escape(tok.strip())
            for tok in backtick_re.findall(body)
            if len(tok.strip()) >= 3 and tok.strip() not in ("src/config/settings.py", "memory_write")
        ]
        if literals:
            pat_str = "(" + "|".join(literals) + ")"
            parsed_rules.append({
                "rule_id": rule_id,
                "pattern": pat_str,
                "description": body,
            })

    return parsed_rules if parsed_rules else default_rules


# ---------------------------------------------------------------------------
# Step Data Model (V2)
# ---------------------------------------------------------------------------

@dataclass
class Step:
    step_index: int
    session_id: str
    task_type: str
    role: str
    text: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    todos: Optional[List[Dict[str, Any]]] = None
    memory_writes: List[str] = field(default_factory=list)
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_total: int = 0
    cumulative_context_tokens: int = 0
    tokens_since_compaction: int = 0
    post_compaction: bool = False
    compaction_index: int = 0
    is_compaction_event: bool = False
    tokens_at_last_compaction: int = 0
    bucket_id: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0
    has_real_cache_telemetry: bool = False
    ts_request: float = 0.0
    ts_response: float = 0.0
    step_confidence: float = 1.0

    def to_record(self) -> Dict[str, Any]:
        return {
            "step_index": int(self.step_index),
            "session_id": str(self.session_id),
            "task_type": str(self.task_type),
            "role": str(self.role),
            "text": str(self.text),
            "tool_calls": json.dumps(self.tool_calls, ensure_ascii=False, sort_keys=True),
            "todos": json.dumps(self.todos, ensure_ascii=False, sort_keys=True) if self.todos is not None else None,
            "memory_writes": json.dumps(self.memory_writes, ensure_ascii=False),
            "tokens_input": int(self.tokens_input),
            "tokens_output": int(self.tokens_output),
            "tokens_total": int(self.tokens_total),
            "cumulative_context_tokens": int(self.cumulative_context_tokens),
            "tokens_since_compaction": int(self.tokens_since_compaction),
            "post_compaction": bool(self.post_compaction),
            "compaction_index": int(self.compaction_index),
            "is_compaction_event": bool(self.is_compaction_event),
            "tokens_at_last_compaction": int(self.tokens_at_last_compaction),
            "bucket_id": int(self.bucket_id),
            "prompt_cache_hit_tokens": int(self.prompt_cache_hit_tokens),
            "prompt_cache_miss_tokens": int(self.prompt_cache_miss_tokens),
            "has_real_cache_telemetry": bool(self.has_real_cache_telemetry),
            "ts_request": float(self.ts_request),
            "ts_response": float(self.ts_response),
            "step_confidence": float(self.step_confidence),
        }


def _safe_json_loads(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        raw_str = raw.strip()
        if not raw_str:
            return {}
        try:
            parsed = json.loads(raw_str)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except Exception:
            return {"text": raw_str}
    return {}


def _extract_timestamp(row_dict: Dict[str, Any], data_dict: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
    for k in keys:
        if k in row_dict and row_dict[k] is not None:
            try:
                val = float(row_dict[k])
                return val / 1000.0 if val > 1e11 else val
            except (ValueError, TypeError):
                pass
        if k in data_dict and data_dict[k] is not None:
            try:
                val = float(data_dict[k])
                return val / 1000.0 if val > 1e11 else val
            except (ValueError, TypeError):
                pass
    return default


def _extract_usage_cache_fields(m_data: Dict[str, Any], msg_parts: List[Dict[str, Any]]) -> Tuple[int, int, bool]:
    """
    Item 3: Extracts real prompt_cache_hit_tokens and prompt_cache_miss_tokens from
    message.data or part.data usage structures when present.
    """
    hit_tokens = 0
    miss_tokens = 0
    found = False

    candidates = [m_data, m_data.get("usage") or {}, m_data.get("tokens") or {}]
    for p in msg_parts:
        pd_obj = p.get("_parsed_data") or {}
        candidates.append(pd_obj)
        if isinstance(pd_obj.get("usage"), dict):
            candidates.append(pd_obj["usage"])
        if isinstance(pd_obj.get("tokens"), dict):
            candidates.append(pd_obj["tokens"])

    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        h = (
            cand.get("prompt_cache_hit_tokens")
            if cand.get("prompt_cache_hit_tokens") is not None
            else cand.get("cache_read_input_tokens", cand.get("cached_tokens", cand.get("cache_hit_tokens")))
        )
        m = (
            cand.get("prompt_cache_miss_tokens")
            if cand.get("prompt_cache_miss_tokens") is not None
            else cand.get("cache_creation_input_tokens", cand.get("uncached_input_tokens", cand.get("cache_miss_tokens")))
        )
        if h is not None or m is not None:
            found = True
            if isinstance(h, (int, float)):
                hit_tokens += int(h)
            if isinstance(m, (int, float)):
                miss_tokens += int(m)
            # Only read once per message if top-level message usage already provided it
            if cand is m_data or cand is m_data.get("usage"):
                break

    return hit_tokens, miss_tokens, found


# ---------------------------------------------------------------------------
# Read OpenCode SQLite & Reconstruct Steps
# ---------------------------------------------------------------------------

def read_opencode_sqlite(
    sqlite_path: str | Path,
    bucket_size_tokens: int = 5000,
) -> List[Step]:
    sqlite_path = Path(sqlite_path).expanduser().resolve()
    if not sqlite_path.exists():
        raise FileNotFoundError(f"OpenCode SQLite database not found at: {sqlite_path}")

    with open_sqlite(sqlite_path, write=False) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r["name"] for r in cursor.fetchall()}
        required = {"session", "message", "part"}
        missing = required - tables
        if missing:
            raise ValueError(f"SQLite DB {sqlite_path} is missing required OpenCode tables: {sorted(missing)}")

        cursor.execute("SELECT * FROM session")
        sessions = {str(dict(r).get("id", "")): dict(r) for r in cursor.fetchall()}

        cursor.execute("SELECT * FROM message")
        messages = [dict(r) for r in cursor.fetchall()]

        cursor.execute("SELECT * FROM part")
        parts = [dict(r) for r in cursor.fetchall()]

    parts_by_msg: Dict[str, List[Dict[str, Any]]] = {}
    for p in parts:
        p_data = _safe_json_loads(p.get("data"))
        msg_id = str(
            p.get("message_id")
            or p.get("messageID")
            or p_data.get("message_id")
            or p_data.get("messageID")
            or ""
        )
        p["_parsed_data"] = p_data
        p["_sort_key"] = (
            _extract_timestamp(p, p_data, ["created_at", "time_created", "index", "sort_order", "id"], 0.0),
            str(p.get("id", "")),
        )
        parts_by_msg.setdefault(msg_id, []).append(p)

    for msg_id in parts_by_msg:
        parts_by_msg[msg_id].sort(key=lambda x: x["_sort_key"])

    msgs_by_session: Dict[str, List[Dict[str, Any]]] = {}
    for m in messages:
        m_data = _safe_json_loads(m.get("data"))
        sess_id = str(
            m.get("session_id")
            or m.get("sessionID")
            or m_data.get("session_id")
            or m_data.get("sessionID")
            or ""
        )
        m["_parsed_data"] = m_data
        req_ts = _extract_timestamp(
            m, m_data, ["ts_request", "request_ts", "created_at", "time_created", "timestamp"], 0.0
        )
        m["_sort_key"] = (req_ts, str(m.get("id", "")))
        msgs_by_session.setdefault(sess_id, []).append(m)

    session_ids = sorted(msgs_by_session.keys())
    all_steps: List[Step] = []

    for sess_id in session_ids:
        sess_msgs = sorted(msgs_by_session[sess_id], key=lambda x: x["_sort_key"])
        sess_meta = _safe_json_loads(sessions.get(sess_id, {}).get("data"))

        # Determine session task_type from session metadata or first user message
        first_user_text = ""
        for m in sess_msgs:
            r_role = str(m.get("role") or m["_parsed_data"].get("role") or "").lower()
            if "user" in r_role:
                for p in parts_by_msg.get(str(m.get("id", "")), []):
                    txt = str(p["_parsed_data"].get("text") or p["_parsed_data"].get("content") or "")
                    if txt:
                        first_user_text += " " + txt
                if first_user_text.strip():
                    break
        task_type = str(sess_meta.get("task_type") or classify_task_type(first_user_text.strip()))

        running_cumulative_tokens = 0
        running_since_compaction = 0
        compaction_index = 0
        tokens_at_last_compaction = 0
        pending_compaction_flag = False
        step_idx = 0

        for m in sess_msgs:
            m_id = str(m.get("id", ""))
            m_data = m["_parsed_data"]
            role = str(m.get("role") or m_data.get("role") or "assistant").lower()

            # Check if this message or its parts represent an explicit compaction checkpoint
            explicit_compact = bool(
                m_data.get("compaction")
                or m_data.get("is_compaction")
                or m_data.get("compacted")
                or role == "compaction"
            )
            msg_parts = parts_by_msg.get(m_id, [])
            for p in msg_parts:
                p_type = str(p.get("type") or p["_parsed_data"].get("type") or "").lower()
                if p_type in ("compaction", "summary_checkpoint") or p["_parsed_data"].get("compaction"):
                    explicit_compact = True

            # Also detect implicit compaction if message reports an explicit input_tokens drop > 40%
            reported_input = m_data.get("input_tokens")
            if (
                not explicit_compact
                and isinstance(reported_input, (int, float))
                and running_since_compaction > 40000
                and reported_input < 0.60 * running_since_compaction
            ):
                explicit_compact = True

            if explicit_compact:
                compaction_index += 1
                tokens_at_last_compaction = int(m_data.get("tokens_at_compaction") or running_since_compaction)
                # Keep-tokens retained after compaction (default 15,000 in OpenCode V2)
                keep_tokens = int(m_data.get("keep_tokens", 15000))
                running_since_compaction = keep_tokens
                pending_compaction_flag = True
                # If the message is purely a system compaction marker with no user/assistant role, continue
                if role == "compaction" or m_data.get("compaction_only"):
                    continue

            if role not in ("assistant", "user"):
                role = "assistant" if "assistant" in role else "user"

            req_ts = _extract_timestamp(
                m, m_data, ["ts_request", "request_ts", "created_at", "time_created", "timestamp"], 0.0
            )
            resp_ts = _extract_timestamp(
                m, m_data, ["ts_response", "response_ts", "updated_at", "time_updated", "completed_at"], req_ts + 1.0
            )
            if resp_ts < req_ts:
                resp_ts = req_ts

            text_fragments: List[str] = []
            tool_calls: List[Dict[str, Any]] = []
            pending_tools_by_id: Dict[str, Dict[str, Any]] = {}
            todos: Optional[List[Dict[str, Any]]] = None
            memory_writes: List[str] = []
            part_tokens_sum = 0

            if not msg_parts and m_data.get("text"):
                txt = str(m_data["text"])
                text_fragments.append(txt)
                part_tokens_sum += count_tokens(txt)

            for p in msg_parts:
                p_data = p["_parsed_data"]
                p_type = str(p.get("type") or p_data.get("type") or "text").lower()

                explicit_tokens = p_data.get("tokens")
                if isinstance(explicit_tokens, (int, float)) and explicit_tokens > 0:
                    part_tokens_sum += int(explicit_tokens)
                else:
                    part_tokens_sum += count_tokens(p_data)

                if p_type in ("text", "reasoning", "message"):
                    txt = str(p_data.get("text") or p_data.get("content") or "")
                    if txt:
                        text_fragments.append(txt)

                elif p_type in ("memory", "memory-write", "memory_write"):
                    mem_txt = str(p_data.get("content") or p_data.get("fact") or p_data.get("text") or "")
                    if mem_txt:
                        memory_writes.append(mem_txt)

                elif p_type in ("tool-call", "tool_call", "tool-invocation", "tool"):
                    call_id = str(p_data.get("call_id") or p_data.get("id") or f"call_{len(tool_calls)}")
                    name = str(p_data.get("name") or p_data.get("tool") or p_data.get("tool_name") or "unknown").lower()
                    params = p_data.get("params") or p_data.get("input") or p_data.get("arguments") or {}
                    if isinstance(params, str):
                        params = _safe_json_loads(params)
                    params_dict = params if isinstance(params, dict) else {"raw": params}

                    # Detect memory writes executed via tool calls
                    target_path = str(params_dict.get("path") or params_dict.get("file_path") or "").lower()
                    if name in ("memory_write", "save_memory", "update_memory", "remember") or "memory" in target_path:
                        mem_payload = str(
                            params_dict.get("content")
                            or params_dict.get("fact")
                            or params_dict.get("new_string")
                            or params_dict.get("text")
                            or ""
                        )
                        if mem_payload:
                            memory_writes.append(mem_payload)

                    tc_entry: Dict[str, Any] = {
                        "call_id": call_id,
                        "name": name,
                        "params": params_dict,
                        "result": p_data.get("result"),
                        "error": p_data.get("error"),
                        "exit_code": p_data.get("exit_code", 0 if p_data.get("error") is None else 1),
                    }
                    if tc_entry["result"] is not None:
                        tc_entry["result_tokens"] = int(
                            p_data.get("result_tokens") or count_tokens(tc_entry["result"])
                        )
                    else:
                        tc_entry["result_tokens"] = 0

                    tool_calls.append(tc_entry)
                    pending_tools_by_id[call_id] = tc_entry

                elif p_type in ("tool-result", "tool_result"):
                    call_id = str(p_data.get("call_id") or p_data.get("tool_call_id") or p_data.get("id") or "")
                    result_val = p_data.get("result") if "result" in p_data else p_data.get("output", "")
                    error_val = p_data.get("error")
                    exit_code = p_data.get("exit_code")
                    if exit_code is None:
                        exit_code = 1 if error_val else 0
                    res_tokens = int(p_data.get("result_tokens") or count_tokens(result_val))

                    if call_id and call_id in pending_tools_by_id:
                        target = pending_tools_by_id[call_id]
                        target["result"] = result_val
                        target["error"] = error_val
                        target["exit_code"] = exit_code
                        target["result_tokens"] = res_tokens
                    elif tool_calls and tool_calls[-1].get("result") is None:
                        target = tool_calls[-1]
                        target["result"] = result_val
                        target["error"] = error_val
                        target["exit_code"] = exit_code
                        target["result_tokens"] = res_tokens
                    else:
                        tool_calls.append({
                            "call_id": call_id or f"call_{len(tool_calls)}",
                            "name": str(p_data.get("name") or p_data.get("tool_name") or "unknown").lower(),
                            "params": p_data.get("params") or {},
                            "result": result_val,
                            "error": error_val,
                            "exit_code": exit_code,
                            "result_tokens": res_tokens,
                        })

                elif p_type in ("todo", "todos", "todowrite"):
                    raw_todos = p_data.get("todos") if "todos" in p_data else p_data.get("items", [])
                    if isinstance(raw_todos, list):
                        todos = raw_todos

            # Token accounting per turn:
            tokens_input = running_since_compaction
            tokens_output = max(1, part_tokens_sum)
            running_cumulative_tokens += tokens_output
            running_since_compaction += tokens_output
            tokens_total = running_since_compaction

            # Item 1: Bucket strictly on tokens_since_compaction
            bucket_id = running_since_compaction // bucket_size_tokens

            # Item 3: Capture real prompt_cache_hit_tokens / prompt_cache_miss_tokens
            cache_hit, cache_miss, has_cache = _extract_usage_cache_fields(m_data, msg_parts)

            is_comp_event = False
            if role == "assistant" and pending_compaction_flag:
                is_comp_event = True
                pending_compaction_flag = False

            step_conf = min(1.0, round(0.4 + 0.12 * min(step_idx, 5), 2))

            step = Step(
                step_index=step_idx,
                session_id=sess_id,
                task_type=task_type,
                role=role,
                text="\n".join(text_fragments).strip(),
                tool_calls=tool_calls,
                todos=todos,
                memory_writes=memory_writes,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                tokens_total=tokens_total,
                cumulative_context_tokens=running_cumulative_tokens,
                tokens_since_compaction=running_since_compaction,
                post_compaction=bool(compaction_index > 0),
                compaction_index=compaction_index,
                is_compaction_event=is_comp_event,
                tokens_at_last_compaction=tokens_at_last_compaction,
                bucket_id=bucket_id,
                prompt_cache_hit_tokens=cache_hit,
                prompt_cache_miss_tokens=cache_miss,
                has_real_cache_telemetry=has_cache,
                ts_request=req_ts,
                ts_response=resp_ts,
                step_confidence=step_conf,
            )
            all_steps.append(step)
            step_idx += 1

    return all_steps


# ---------------------------------------------------------------------------
# Persistence to DuckDB
# ---------------------------------------------------------------------------

def save_steps_to_duckdb(
    steps: List[Step],
    duckdb_path: str | Path,
    agents_rules: Optional[List[Dict[str, str]]] = None,
) -> pd.DataFrame:
    duckdb_path = Path(duckdb_path).expanduser().resolve()
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    records = [s.to_record() for s in steps]
    df = pd.DataFrame(records) if records else pd.DataFrame(columns=[
        "step_index", "session_id", "task_type", "role", "text", "tool_calls", "todos", "memory_writes",
        "tokens_input", "tokens_output", "tokens_total", "cumulative_context_tokens",
        "tokens_since_compaction", "post_compaction", "compaction_index", "is_compaction_event",
        "tokens_at_last_compaction", "bucket_id", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
        "has_real_cache_telemetry", "ts_request", "ts_response", "step_confidence"
    ])

    rules_df = pd.DataFrame(agents_rules or parse_agents_md_rules())

    with open_duckdb(duckdb_path, read_only=False) as con:
        con.execute("DROP TABLE IF EXISTS steps")
        con.execute("""
            CREATE TABLE steps (
                step_index BIGINT,
                session_id TEXT,
                task_type TEXT,
                role TEXT,
                text TEXT,
                tool_calls JSON,
                todos JSON,
                memory_writes JSON,
                tokens_input BIGINT,
                tokens_output BIGINT,
                tokens_total BIGINT,
                cumulative_context_tokens BIGINT,
                tokens_since_compaction BIGINT,
                post_compaction BOOLEAN,
                compaction_index BIGINT,
                is_compaction_event BOOLEAN,
                tokens_at_last_compaction BIGINT,
                bucket_id BIGINT,
                prompt_cache_hit_tokens BIGINT,
                prompt_cache_miss_tokens BIGINT,
                has_real_cache_telemetry BOOLEAN,
                ts_request DOUBLE,
                ts_response DOUBLE,
                step_confidence DOUBLE
            )
        """)
        if not df.empty:
            con.register("df_steps_view", df)
            con.execute("""
                INSERT INTO steps
                SELECT
                    step_index, session_id, task_type, role, text,
                    tool_calls, todos, memory_writes,
                    tokens_input, tokens_output, tokens_total,
                    cumulative_context_tokens, tokens_since_compaction,
                    post_compaction, compaction_index, is_compaction_event,
                    tokens_at_last_compaction, bucket_id,
                    prompt_cache_hit_tokens, prompt_cache_miss_tokens,
                    has_real_cache_telemetry, ts_request, ts_response, step_confidence
                FROM df_steps_view
            """)

        con.execute("DROP TABLE IF EXISTS agents_rules")
        con.register("df_rules_view", rules_df)
        con.execute("CREATE TABLE agents_rules AS SELECT * FROM df_rules_view")

    return df


def load_steps_from_duckdb(duckdb_path: str | Path) -> pd.DataFrame:
    with open_duckdb(duckdb_path, read_only=True) as con:
        df = con.execute(
            "SELECT * FROM steps ORDER BY session_id ASC, step_index ASC"
        ).df()
    return df


def load_agents_rules_from_duckdb(duckdb_path: str | Path) -> List[Dict[str, str]]:
    try:
        with open_duckdb(duckdb_path, read_only=True) as con:
            df = con.execute("SELECT * FROM agents_rules").df()
        if not df.empty:
            return df.to_dict("records")
    except Exception:
        pass
    return parse_agents_md_rules()


def run_ingestion(
    sqlite_path: str | Path,
    duckdb_path: str | Path,
    config_path: Optional[str | Path] = None,
    agents_md_path: Optional[str | Path] = "AGENTS.md",
) -> pd.DataFrame:
    bucket_size = 5000
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            bucket_size = int(cfg.get("bucket_size_tokens", 5000))

    rules = parse_agents_md_rules(agents_md_path)
    steps = read_opencode_sqlite(sqlite_path, bucket_size_tokens=bucket_size)
    df = save_steps_to_duckdb(steps, duckdb_path, agents_rules=rules)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Ingest OpenCode SQLite DB into DuckDB steps table.")
    parser.add_argument(
        "--db-path",
        type=str,
        default=os.path.expanduser("~/.local/share/opencode/opencode.db"),
        help="Path to OpenCode SQLite database (opencode.db)",
    )
    parser.add_argument("--out-duckdb", type=str, default="data/steps.duckdb", help="Output DuckDB path")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--agents-md", type=str, default="AGENTS.md", help="Path to AGENTS.md rules file")
    args = parser.parse_args()

    df = run_ingestion(args.db_path, args.out_duckdb, args.config, args.agents_md)
    n_sessions = df["session_id"].nunique() if not df.empty else 0
    max_since = int(df["tokens_since_compaction"].max()) if not df.empty else 0
    max_cum = int(df["cumulative_context_tokens"].max()) if not df.empty else 0
    print(f"[Phase 1: Ingestion] Tokenizer backend: {TOKENIZER_BACKEND}")
    print(
        f"[Phase 1: Ingestion] Ingested {len(df)} steps across {n_sessions} sessions "
        f"(max tokens_since_compaction={max_since:,}, max cumulative={max_cum:,}) -> {args.out_duckdb}"
    )


if __name__ == "__main__":
    main()
