# Signal Specifications V2 (`docs/SIGNALS.md`)

Total reliability weight sum: **$W = 14.8$** across 15 signals (17 evaluators: `s1`–`s13`, `s14a`, `s14b`, `s14c`, `s15`).

| ID | Signal Name | Tier | Weight | State Type | Exact Firing Rule |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **s13** | **`AGENTS.md` Retrieval Failure** | Arch #1 | **`1.5`** | Rule-Matcher | Parses `AGENTS.md` bullets into regex rules at ingestion time; fires when tool calls or arguments violate any forbidden rule. |
| **s15** | **Compaction Recovery Failure** | Arch #2 | **`1.3`** | Compaction-Epoch | Fires when `recovery_ratio = post_compaction_rot_score / pre_compaction_rot_score > 0.65` (and `post_rot_score >= 180`). |
| **s14c** | **Lost Memory Write (Gate)** | Arch #3 | **`1.2`** | Cross-Compaction | Extracts candidate facts (`"note that X"`, `"remember X"`, `"important: X"`) from pre-compaction assistant text; fires if missing from post-compaction memory writes. |
| **s1** | **Tool Error** | Tier 1 | **`1.3`** | Stateless | Any tool call in the step has `error` non-null/non-empty OR `exit_code != 0`. |
| **s2** | **Edit Failure** | Tier 1 | **`1.3`** | Stateless | `tool_name` in `{edit, write, patch, ...}` AND result/error matches `/string not found\|no match\|conflict\|.../i`. |
| **s3** | **Retry Loop** | Tier 1 | **`1.3`** | Windowed (3) | `sha256(tool_name + canonical_json(params))` matches a failed tool call (`s1 == True`) within the last 3 assistant steps. |
| **s4** | **Session Re-Read** | Tier 1 | **`1.3`** | Session-Stateful | `tool_name` in `{read, read_file, view_file, cat}` AND file `path` was already read earlier in the session. |
| **s5** | **Self-Correction** | Tier 2 | **`0.8`** | Stateless | Assistant text matches `/\b(I apologi[sz]e\|let me fix\|my mistake\|...)\b/i`. Auto-false on tool-call-only steps. |
| **s6** | **Duplicate Read (Window)** | Tier 2 | **`0.8`** | Windowed (5) | `tool_name` in `{read, ...}` AND file `path` was read within the last 5 assistant steps. |
| **s7** | **Unreferenced Large Output** | Tier 3 | **`0.6`** | Lookahead (3) | Tool result has `result_tokens > 5000` AND none of the next 3 assistant steps contain any 80-char shingle from that output. |
| **s8** | **Repetition Entropy Drop** | Tier 3 | **`0.6`** | Windowed (20) | Shannon entropy $H$ of last-20 tool-name sequence is `< H_baseline - 0.3`. |
| **s11** | **Contradictory Edit** | Tier 3 | **`0.6`** | Session-Stateful | Edit reverts `path` to hash from 2 edits ago OR `path` edited $\ge 3$ times in 20 steps. |
| **s14a** | **Memory Write Latency** | Memory | **`0.5`** | Post-Compaction | Number of assistant steps from compaction event to first `memory_write` exceeds `memory_write_max_latency_steps` (3). |
| **s14b** | **Memory Write Burst / Starve** | Memory | **`0.5`** | Post-Compaction | Count of `memory_write` calls in first 10 post-compaction steps is `0` (when candidate facts exist) or $\ge 5$ (thrashing burst). |
| **s9** | **Instruction Amnesia** | Defended | **`0.3`** | Session-Stateful | Violates negative constraint from user messages (down-weighted to `0.3` as `AGENTS.md` + memory defend this layer). |
| **s10** | **Stale Todo Snapshot** | Defended | **`0.3`** | Session-Stateful | `sha256(todos)` matches snapshot $\ge 10$ steps ago, absent in last 5 (down-weighted to `0.3`). |
| **s12** | **Latency Outlier** | Side | **`0.3`** | Session-Stateful | Turn response duration `(ts_response - ts_request) > 3.0 * session_median_latency`. |
