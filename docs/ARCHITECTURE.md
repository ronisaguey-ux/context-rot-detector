# Architecture & Data Model V2 (`docs/ARCHITECTURE.md`)

## 1. Key Tables in `data/steps.duckdb`

### `steps` Table
- `step_index` (`BIGINT`), `session_id` (`TEXT`), `task_type` (`TEXT`: `refactor` / `bugfix` / `feature` / `test` / `explore`), `role` (`TEXT`)
- `cumulative_context_tokens` (`BIGINT`): Raw cumulative tokens from cold start of the session.
- `tokens_since_compaction` (`BIGINT`): Context fill accumulated since the most recent compaction checkpoint (or session start if `compaction_index == 0`).
- `post_compaction` (`BOOLEAN`), `compaction_index` (`BIGINT`), `is_compaction_event` (`BOOLEAN`), `tokens_at_last_compaction` (`BIGINT`).
- `bucket_id` (`BIGINT`): Strictly `floor(tokens_since_compaction / 5000)`.
- `prompt_cache_hit_tokens` (`BIGINT`), `prompt_cache_miss_tokens` (`BIGINT`), `has_real_cache_telemetry` (`BOOLEAN`).

### `compaction_events` Table (Item 8)
One row per compaction event across all sessions:
- `session_id`, `task_type`, `compaction_index`, `step_index`
- `tokens_at_compaction` (`BIGINT`): Fill level (`tokens_since_compaction`) immediately prior to compaction.
- `pre_rot_score` (`DOUBLE`), `post_rot_score` (`DOUBLE`), `recovery_ratio` (`DOUBLE` = `post_rot_score / pre_rot_score`).
- `memory_writes_after` (`BIGINT`, Signal 14b), `memory_write_latency_steps` (`BIGINT`, Signal 14a).
- `candidate_facts_count` (`BIGINT`), `lost_candidates_count` (`BIGINT`), `s14c_lost_state` (`BOOLEAN`, Signal 14c), `s15_failed_recovery` (`BOOLEAN`, Signal 15).

### `agents_rules` Table (Item 5)
Discrete regex rules parsed from `AGENTS.md` at ingestion time and evaluated against every assistant turn's tool calls (`Signal 13`).
