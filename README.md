# Context Rot Detector V2 (Signals 1–15, `tokens_since_compaction` Buckets, 900k Window)

An architecture-calibrated behavioral telemetry and compaction-threshold optimizer for **OpenCode** (`opencode.db`), tailored for **900k-context-window, `AGENTS.md`-injected, memory-gated** setups.

## What's New in V2 (Priorities 1–4)

1. **`tokens_since_compaction` Bucketing (P1)**: `steps.duckdb` tracks `post_compaction: bool`, `compaction_index: int`, `is_compaction_event: bool`, `tokens_at_last_compaction: int`, and `tokens_since_compaction: int`. Buckets are computed strictly as `floor(tokens_since_compaction / 5000)` so a step at 300k after compaction is never confounded with 300k from cold start.
2. **Architecture-Calibrated Signal Weights (`W = 14.8`, P1)**:
   - Up-weights Tier 1 direct failures (`s1`–`s4`) to **`1.3`**.
   - Down-weights `s9` (instruction amnesia) and `s10` (stale todos) to **`0.3`** (already defended by `AGENTS.md` + memory).
   - Adds **Signal 13 (`s13`, weight `1.5`)**: `AGENTS.md` retrieval failure (parses `AGENTS.md` markdown bullets into discrete regex matchers at ingestion time and flags violations in tool calls).
   - Adds **Signals 14a (`0.5`), 14b (`0.5`), and 14c (`1.2`)**: Memory-gating telemetry (`14a` = steps from compaction to first memory write; `14b` = memory write burst/starvation in first 10 post-compaction steps; `14c` = pre-compaction candidate facts `"note that X"` / `"remember X"` missing in post-compaction memory).
   - Adds **Signal 15 (`s15`, weight `1.3`) + `compaction_events` table**: Computes `recovery_ratio = post_compaction_rot_score / pre_compaction_rot_score` per compaction event and locates the **Compaction Recovery Knee** vs. `tokens_at_compaction`.
3. **Real Prompt-Cache Telemetry & Configurable Pricing (P1 & P4)**: Captures `prompt_cache_hit_tokens` and `prompt_cache_miss_tokens` from `message.data` / `part.data` (`usage` fields) and computes real USD cost using `price_cached_input_per_mtok`, `price_uncached_input_per_mtok`, and `price_output_per_mtok` in `config.yaml` (using simulation only as a fallback when telemetry is absent).
4. **Noise-Resilient Hard-Failure Floor ($\ge 2$ Fires, P1)**: Requires `(s1_fires + s2_fires) >= 2` (`hard_failure_min_fires: 2`) before flooring a bucket at `500`, preventing single transient tool errors from dominating the curve.
5. **Analysis & Guard Fixes (P3)**:
   - **`--min-fill-tokens` Guard**: Refuses to report false early onset on short/truncated datasets and reports both max `tokens_since_compaction` and max raw `cumulative_context_tokens`.
   - **Signal 14c Post-Compaction Stratification**: Splits post-compaction buckets into `post_compaction_saved_ok` (`s14c=False`) vs `post_compaction_lost_state` (`s14c=True`) in `outputs/score_curve_stratified.csv` and `outputs/report.html`.
   - **`n_steps`-Weighted Aggregation & Confidence Column**: Weights bucket curves by `n_steps`, exports `confidence = min(1.0, n_steps / 10)` in `outputs/bucket_scores.csv`, and flags `n_steps < 10` as `low_confidence`.
   - **Coarse `task_type` Tagging**: Tags each session from its opening user message (`refactor`, `bugfix`, `feature`, `test`, `explore`) and outputs stratified curves (`outputs/task_type_curves.csv`).
6. **Inversion-Safe OpenCode Config Output (P4)**: Verified against [OpenCode V2 Compaction Docs](https://opencode.ai/v2/docs/compaction/): OpenCode uses `compaction.buffer` (tokens kept free *below* the model context limit), so `buffer = model_context_window - threshold_T` (`900,000 - T`), preventing catastrophic inversion.

## Quick Start

```bash
# Run full V2 pipeline on your real OpenCode SQLite DB
python3 run_pipeline.py --db-path ~/.local/share/opencode/opencode.db --agents-md AGENTS.md

# Or run on the included 900k-window multi-compaction sample DB
python3 run_pipeline.py

# Run unit tests
python3 tests/test_pipeline.py -v
```
