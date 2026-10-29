#!/usr/bin/env python3
"""
Synthetic 900k-Window OpenCode SQLite Generator (generate_sample_db.py) [V2]

Generates a realistic 16-session OpenCode SQLite database (`data/sample_opencode.db`)
designed for a 900k-context-window, AGENTS.md-injected, memory-gated architecture:
  - Sessions reach 420k to 550k `tokens_since_compaction` and up to 980k raw
    `cumulative_context_tokens` across 1-2 compaction events per session.
  - Every message includes realistic `usage.prompt_cache_hit_tokens` and
    `usage.prompt_cache_miss_tokens` telemetry.
  - Opening user prompts cover all 5 coarse task types (`refactor`, `bugfix`,
    `feature`, `test`, `explore`).
  - Models pre-compaction candidate facts ("Note that ...", "Important: ...")
    and `memory_write` tool calls across compaction boundaries:
      * Compactions at <= 250k fill save state cleanly (`s14c=False`, low `recovery_ratio` ~0.25)
      * Compactions at >= 275k fill suffer memory-gate loss (`s14c=True`, `s14a=True`,
        high `recovery_ratio` > 0.70 triggering `s15=True`)
  - Past ~260k-280k `tokens_since_compaction`, AGENTS.md violations (`s13`) and
    Tier 1 failures (`s1`-`s4`) accelerate sharply.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ingest import open_sqlite


FILES_POOL = [
    "src/auth/jwt_handler.py",
    "src/db/connection.py",
    "src/api/routes.py",
    "src/models/user.py",
    "src/services/billing.py",
    "src/utils/validators.py",
    "src/cache/redis_store.py",
    "src/worker/queue.py",
    "tests/test_auth.py",
    "tests/test_billing.py",
]

DIVERSE_TOOLS = ["read", "grep", "glob", "edit", "bash", "write", "list_dir", "search_symbol"]

TASK_OPENERS: Dict[str, str] = {
    "refactor": "Please refactor and modularize the authentication and billing services cleanly. Never modify legacy_config.py and don't use deprecated_crypto.",
    "bugfix":   "Fix the intermittent token regression crash in the session renewal handler. Never modify legacy_config.py and don't use deprecated_crypto.",
    "feature":  "Implement the new multi-tenant webhook delivery endpoint and retry queue. Never modify legacy_config.py and don't use deprecated_crypto.",
    "test":     "Write comprehensive unit tests and integration test coverage for billing webhooks. Never modify legacy_config.py and don't use deprecated_crypto.",
    "explore":  "Explore and audit the codebase architecture to trace how session checkpoints work. Never modify legacy_config.py and don't use deprecated_crypto.",
}


def generate_sample_opencode_db(db_path: str | Path = "data/sample_opencode.db", seed: int = 2026) -> Path:
    rng = random.Random(seed)
    db_path = Path(db_path).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    # 16 sessions spanning 5 task types, with compactions at varying fill levels (175k to 380k)
    # and post-compaction runs reaching up to 500k tokens_since_compaction (~900k+ cumulative)
    sessions_spec: List[Tuple[str, str, int, int, bool]] = [
        # (session_id, task_type, compact_at_tokens, post_compact_target_tokens, is_offrails)
        ("sess_01_refactor",         "refactor", 200000, 420000, False),
        ("sess_02_bugfix",           "bugfix",   225000, 440000, False),
        ("sess_03_offrails_refactor","refactor", 325000, 500000, True),
        ("sess_04_feature",          "feature",  250000, 450000, False),
        ("sess_05_test",             "test",     180000, 410000, False),
        ("sess_06_explore",          "explore",  240000, 430000, False),
        ("sess_07_offrails_bugfix",  "bugfix",   350000, 500000, True),
        ("sess_08_feature",          "feature",  275000, 450000, False),
        ("sess_09_refactor",         "refactor", 210000, 430000, False),
        ("sess_10_test",             "test",     295000, 460000, False),
        ("sess_11_offrails_feature", "feature",  360000, 500000, True),
        ("sess_12_bugfix",           "bugfix",   230000, 440000, False),
        ("sess_13_explore",          "explore",  310000, 460000, False),
        ("sess_14_offrails_refactor","refactor", 375000, 500000, True),
    ]

    with open_sqlite(db_path, write=True) as conn:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE session (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at REAL,
                updated_at REAL,
                data TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE message (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                role TEXT,
                created_at REAL,
                updated_at REAL,
                data TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE part (
                id TEXT PRIMARY KEY,
                message_id TEXT,
                session_id TEXT,
                type TEXT,
                created_at REAL,
                data TEXT
            )
        """)

        base_epoch = 1760000000.0
        msg_counter = 0
        part_counter = 0

        for s_idx, (sess_id, task_type, compact_at, post_target, is_offrails) in enumerate(sessions_spec):
            sess_start = base_epoch + s_idx * 25000.0
            cur.execute(
                "INSERT INTO session VALUES (?, ?, ?, ?, ?)",
                (
                    sess_id,
                    f"Session {sess_id} ({task_type})",
                    sess_start,
                    sess_start + 14400.0,
                    json.dumps({
                        "off_rails": is_offrails,
                        "task_type": task_type,
                        "model": "deepseek-v3",
                        "context_window": 900000,
                    }),
                ),
            )

            initial_todos = [
                {"id": "todo_1", "task": f"Execute {task_type} plan", "status": "in_progress"},
                {"id": "todo_2", "task": "Verify AGENTS.md compliance", "status": "pending"},
            ]

            # Candidate facts emitted during Epoch 0 (before compaction)
            fact_1 = f"auth_epoch_salt_key={sess_id}_v2"
            fact_2 = f"retry_backoff_ceiling_ms={1500 + s_idx * 100}"

            cur_ts = sess_start
            turn_idx = 0

            # Run 2 epochs per session: Epoch 0 (0 -> compact_at) and Epoch 1 (15k -> post_target)
            for epoch_idx, target_fill in [(0, compact_at), (1, post_target)]:
                tokens_since_comp = 0 if epoch_idx == 0 else 15000
                pending_retry_call: Dict[str, Any] | None = None
                late_compaction = compact_at >= 275000

                # If entering Epoch 1, emit the compaction checkpoint on the first assistant turn
                first_turn_in_epoch1 = (epoch_idx == 1)
                epoch_step = 0

                while tokens_since_comp < target_fill:
                    bucket_id = tokens_since_comp // 5000  # 0 .. 100 (0k .. 500k)

                    # Emit user message at start of session or every 18 steps
                    if epoch_idx == 0 and epoch_step == 0:
                        msg_counter += 1
                        u_msg_id = f"msg_{msg_counter:06d}"
                        u_text = TASK_OPENERS[task_type]
                        u_tok = 250
                        cur.execute(
                            "INSERT INTO message VALUES (?, ?, ?, ?, ?, ?)",
                            (
                                u_msg_id,
                                sess_id,
                                "user",
                                cur_ts,
                                cur_ts + 0.4,
                                json.dumps({
                                    "role": "user",
                                    "usage": {"prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": u_tok},
                                }),
                            ),
                        )
                        part_counter += 1
                        cur.execute(
                            "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
                            (
                                f"part_{part_counter:06d}",
                                u_msg_id,
                                sess_id,
                                "text",
                                cur_ts,
                                json.dumps({"type": "text", "text": u_text, "tokens": u_tok}),
                            ),
                        )
                        tokens_since_comp += u_tok
                        cur_ts += 1.0

                    # Each assistant step adds ~2,450 tokens (~2 steps per 5k bucket per session -> ~35-55 steps per bucket across corpus)
                    step_token_budget = rng.randint(2350, 2550)
                    fill_k = tokens_since_comp // 1000

                    # Rot onset dynamics:
                    # - Below 255k tokens_since_compaction (bucket < 51): pristine/healthy in pre-compaction & saved-state epochs
                    # - In post-compaction epochs where s14c lost state (late_compaction), moderate degradation starts earlier (~180k)
                    # - 260k - 300k tokens_since_compaction (buckets 52-60): Sharp Rot Knee Onset!
                    # - > 300k tokens_since_compaction (bucket >= 60): Severe context rot
                    near_pre_compact = (epoch_idx == 0 and tokens_since_comp >= 235000 and (target_fill - tokens_since_comp) <= 22000)
                    if near_pre_compact:
                        # Pre-compaction fatigue in the final ~8 steps before compaction
                        rot_prob = 0.42 if not late_compaction else 0.68
                    elif epoch_idx == 1 and epoch_step < 10:
                        # First 10 steps post-compaction: clean recovery if compact_at <= 250k, failed recovery if compact_at >= 275k
                        rot_prob = (0.72 if is_offrails else 0.56) if late_compaction else 0.0
                    elif fill_k < 260:
                        if epoch_idx == 1 and late_compaction and fill_k >= 180:
                            # Stratified 14c curve separation: lost-state sessions degrade earlier than saved-state sessions
                            rot_prob = 0.18 + 0.0022 * (fill_k - 180)
                        else:
                            rot_prob = 0.002 if not is_offrails else 0.005
                    elif fill_k < 305:
                        # Knee region (255k - 305k)
                        rot_prob = 0.22 + 0.009 * (fill_k - 255)
                        if is_offrails:
                            rot_prob += 0.24
                    else:
                        # Deep rot (>305k - 500k)
                        if is_offrails:
                            rot_prob = min(0.98, 0.82 + 0.001 * (fill_k - 305))
                        else:
                            rot_prob = min(0.48, 0.32 + 0.0008 * (fill_k - 305))

                    is_rotten_step = (turn_idx >= 6) and (rng.random() < rot_prob)

                    msg_counter += 1
                    a_msg_id = f"msg_{msg_counter:06d}"

                    latency = 2.2 + rng.uniform(-0.4, 0.5)
                    if is_rotten_step and fill_k >= 260 and rng.random() < (0.45 if is_offrails else 0.20):
                        latency = rng.uniform(8.8, 15.0)

                    req_ts = cur_ts
                    resp_ts = cur_ts + latency
                    cur_ts = resp_ts + 0.4

                    # Real prompt cache telemetry per turn:
                    # On normal turns, ~92% of tokens_since_comp is cache hit; on compaction event turn, cache miss spikes!
                    if first_turn_in_epoch1:
                        cache_hit = 4000
                        cache_miss = tokens_since_comp + step_token_budget + 22000
                    else:
                        cache_hit = int(tokens_since_comp * 0.92)
                        cache_miss = int(tokens_since_comp * 0.08) + step_token_budget

                    msg_meta: Dict[str, Any] = {
                        "role": "assistant",
                        "ts_request": req_ts,
                        "ts_response": resp_ts,
                        "usage": {
                            "prompt_cache_hit_tokens": cache_hit,
                            "prompt_cache_miss_tokens": cache_miss,
                            "output_tokens": step_token_budget,
                        },
                    }
                    if first_turn_in_epoch1:
                        msg_meta["compaction"] = True
                        msg_meta["tokens_at_compaction"] = compact_at
                        msg_meta["keep_tokens"] = 15000
                        first_turn_in_epoch1 = False

                    cur.execute(
                        "INSERT INTO message VALUES (?, ?, ?, ?, ?, ?)",
                        (a_msg_id, sess_id, "assistant", req_ts, resp_ts, json.dumps(msg_meta)),
                    )

                    # Build assistant text (including candidate facts in Epoch 0 for Signal 14c)
                    if epoch_idx == 0 and epoch_step == 4:
                        a_text = f"Note that {fact_1} and remember that {fact_2} for the checkpoint."
                    elif is_rotten_step and rng.random() < 0.55:
                        a_text = f"I apologize for that mistake — let me fix the edit target at turn {turn_idx}."
                    else:
                        a_text = f"Executing {task_type} step {turn_idx} cleanly."

                    if fill_k < 250:
                        a_text += " [BENCHMARK_SHINGLE_REFERENCE_BLOCK_0123456789_ABCDEFGHIJKLMNOPQRSTUVWXYZ_HEALTHY_CHECK_OK]"

                    part_counter += 1
                    cur.execute(
                        "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            f"part_{part_counter:06d}",
                            a_msg_id,
                            sess_id,
                            "text",
                            req_ts + 0.1,
                            json.dumps({"type": "text", "text": a_text, "tokens": 600}),
                        ),
                    )

                    # Memory write behavior after compaction (Signals 14a, 14b, 14c):
                    # - If compact_at <= 250k (healthy compaction): writes both fact_1 and fact_2 to memory at epoch_step 1 (latency = 1 <= 3, s14a=False, s14b=False, s14c=False)
                    # - If compact_at >= 275k (late compaction): delays memory write until epoch_step 6 (>3 -> s14a=True) or bursts/omits fact_2 (s14c=True, s14b=True)
                    if epoch_idx == 1:
                        if not late_compaction and epoch_step in (1, 2):
                            part_counter += 1
                            cur.execute(
                                "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
                                (
                                    f"part_{part_counter:06d}",
                                    a_msg_id,
                                    sess_id,
                                    "tool-call",
                                    req_ts + 0.15,
                                    json.dumps({
                                        "type": "tool-call",
                                        "call_id": f"mem_{msg_counter}",
                                        "name": "memory_write",
                                        "params": {"path": "MEMORY.md", "content": f"Saved checkpoint: {fact_1} and {fact_2}"},
                                        "result": "ok",
                                        "error": None,
                                        "exit_code": 0,
                                        "tokens": 50,
                                    }),
                                ),
                            )
                        elif late_compaction and epoch_step == 6:
                            # Late write (step 6 > 3 -> s14a=True) and omits fact_2 (-> s14c=True) + bursts 5 writes (-> s14b=True)
                            for b_i in range(5):
                                part_counter += 1
                                cur.execute(
                                    "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
                                    (
                                        f"part_{part_counter:06d}",
                                        a_msg_id,
                                        sess_id,
                                        "tool-call",
                                        req_ts + 0.15 + b_i * 0.01,
                                        json.dumps({
                                            "type": "tool-call",
                                            "call_id": f"mem_burst_{msg_counter}_{b_i}",
                                            "name": "memory_write",
                                            "params": {"path": "MEMORY.md", "content": f"Partial stale state {b_i}: {fact_1}"},
                                            "result": "ok",
                                            "error": None,
                                            "exit_code": 0,
                                            "tokens": 20,
                                        }),
                                    ),
                                )

                    # Main tool call(s)
                    tool_tokens = step_token_budget - 600
                    if not is_rotten_step:
                        t_name = DIVERSE_TOOLS[turn_idx % len(DIVERSE_TOOLS)]
                        f_path = f"src/mod_{turn_idx}/comp_{turn_idx}.py"
                        t_params: Dict[str, Any] = {"path": f_path}
                        if t_name == "edit":
                            t_params["new_string"] = f"def fn_{turn_idx}(): return True\n"
                        t_result = f"Success executing {t_name} on {f_path}"
                        t_error = None
                        t_exit = 0
                        res_tok = max(200, tool_tokens)
                    else:
                        if pending_retry_call is not None:
                            t_name = pending_retry_call["name"]
                            t_params = pending_retry_call["params"]
                            t_result = "Error: string not found in target file (retry failed)"
                            t_error = "EditConflictError: string not found"
                            t_exit = 1
                            res_tok = max(200, tool_tokens)
                            pending_retry_call = None
                        else:
                            mode = rng.choice(["agents_md_violation", "edit_fail_double", "reread_dup", "large_unreferenced"])
                            if mode == "agents_md_violation" or (is_offrails and fill_k >= 270 and rng.random() < 0.55):
                                # Signal 13 (AGENTS.md retrieval failure) + Signal 9 + Signal 1 + Signal 2
                                t_name = "edit"
                                t_params = {
                                    "path": "legacy_config.py",
                                    "old_string": "missing_anchor",
                                    "new_string": "import deprecated_crypto\nDROP TABLE users;\n",
                                }
                                t_result = "Error: string not found in legacy_config.py"
                                t_error = "EditError: string not found"
                                t_exit = 1
                                res_tok = max(200, tool_tokens)
                                pending_retry_call = {"name": t_name, "params": t_params}
                            elif mode == "edit_fail_double":
                                t_name = "edit"
                                revert_str = "def state_v1(): pass" if (turn_idx % 2 == 0) else "def state_v2(): pass"
                                t_params = {
                                    "path": FILES_POOL[0],
                                    "old_string": "non_existent_target()",
                                    "new_string": revert_str,
                                }
                                t_result = "Error: string not found in src/auth/jwt_handler.py (0 occurrences)"
                                t_error = "EditError: string not found"
                                t_exit = 1
                                res_tok = max(200, tool_tokens)
                                pending_retry_call = {"name": t_name, "params": t_params}
                            elif mode == "reread_dup":
                                t_name = "read"
                                t_params = {"path": "src/mod_1/comp_1.py"}
                                t_result = "Contents of comp_1.py..."
                                t_error = None
                                t_exit = 0
                                res_tok = max(200, tool_tokens)
                            else:
                                t_name = "read"
                                t_params = {"path": "src/mod_1/comp_1.py"}
                                t_result = (
                                    f"LARGE_DUMP_SESSION_{sess_id}_TURN_{turn_idx}_"
                                    + ("X987654321_UNREFERENCED_LOG_SEGMENT_" * 6)
                                )
                                t_error = None
                                t_exit = 0
                                res_tok = 5400

                    call_id = f"call_{msg_counter}_1"
                    part_counter += 1
                    cur.execute(
                        "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            f"part_{part_counter:06d}",
                            a_msg_id,
                            sess_id,
                            "tool-call",
                            req_ts + 0.2,
                            json.dumps({
                                "type": "tool-call",
                                "call_id": call_id,
                                "name": t_name,
                                "params": t_params,
                                "tokens": tool_tokens // 2,
                            }),
                        ),
                    )
                    part_counter += 1
                    cur.execute(
                        "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            f"part_{part_counter:06d}",
                            a_msg_id,
                            sess_id,
                            "tool-result",
                            req_ts + 0.35,
                            json.dumps({
                                "type": "tool-result",
                                "call_id": call_id,
                                "name": t_name,
                                "result": t_result,
                                "error": t_error,
                                "exit_code": t_exit,
                                "result_tokens": res_tok,
                                "tokens": tool_tokens - (tool_tokens // 2),
                            }),
                        ),
                    )

                    # In deep rot, emit a second failing edit tool call in the same step so >=2 hard failures trigger the 500 floor
                    if is_rotten_step and fill_k >= 265 and (is_offrails or rng.random() < 0.55):
                        call2_id = f"call_{msg_counter}_2"
                        part_counter += 1
                        cur.execute(
                            "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
                            (
                                f"part_{part_counter:06d}",
                                a_msg_id,
                                sess_id,
                                "tool-call",
                                req_ts + 0.42,
                                json.dumps({
                                    "type": "tool-call",
                                    "call_id": call2_id,
                                    "name": "edit",
                                    "params": {"path": "migrations/locked/001_init.sql", "new_string": "DROP TABLE sessions;"},
                                    "result": "Error: no match found in migrations/locked/001_init.sql",
                                    "error": "PatchError: no match",
                                    "exit_code": 1,
                                    "tokens": 30,
                                }),
                            ),
                        )

                    # Also emit repeated 'read' calls in deep rot to drop 20-tool Shannon entropy (Signal 8)
                    if is_rotten_step and fill_k >= 260:
                        for extra_i in range(2):
                            part_counter += 1
                            cur.execute(
                                "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
                                (
                                    f"part_{part_counter:06d}",
                                    a_msg_id,
                                    sess_id,
                                    "tool-call",
                                    req_ts + 0.48 + extra_i * 0.02,
                                    json.dumps({
                                        "type": "tool-call",
                                        "call_id": f"call_{msg_counter}_read_{extra_i}",
                                        "name": "read",
                                        "params": {"path": "src/mod_1/comp_1.py"},
                                        "result": "Cached content",
                                        "error": None,
                                        "exit_code": 0,
                                        "tokens": 10,
                                    }),
                                ),
                            )

                    # Signal 10: Stale todo snapshot
                    if turn_idx in (1, 2):
                        part_counter += 1
                        cur.execute(
                            "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
                            (
                                f"part_{part_counter:06d}",
                                a_msg_id,
                                sess_id,
                                "todo",
                                req_ts + 0.55,
                                json.dumps({"type": "todo", "todos": initial_todos, "tokens": 20}),
                            ),
                        )
                    elif turn_idx == 6:
                        part_counter += 1
                        cur.execute(
                            "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
                            (
                                f"part_{part_counter:06d}",
                                a_msg_id,
                                sess_id,
                                "todo",
                                req_ts + 0.55,
                                json.dumps({"type": "todo", "todos": [{"id": "todo_2", "status": "done"}], "tokens": 20}),
                            ),
                        )
                    elif is_rotten_step and turn_idx >= 16 and (turn_idx % 6 == 0):
                        part_counter += 1
                        cur.execute(
                            "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
                            (
                                f"part_{part_counter:06d}",
                                a_msg_id,
                                sess_id,
                                "todo",
                                req_ts + 0.55,
                                json.dumps({"type": "todo", "todos": initial_todos, "tokens": 20}),
                            ),
                        )

                    tokens_since_comp += step_token_budget
                    turn_idx += 1
                    epoch_step += 1

    return db_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate realistic 900k-window OpenCode SQLite DB.")
    parser.add_argument("--out-db", type=str, default="data/sample_opencode.db", help="Output SQLite DB path")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    args = parser.parse_args()

    out = generate_sample_opencode_db(args.out_db, seed=args.seed)
    print(f"[Sample Generator V2] Created 900k-window OpenCode SQLite DB at: {out}")


if __name__ == "__main__":
    main()
