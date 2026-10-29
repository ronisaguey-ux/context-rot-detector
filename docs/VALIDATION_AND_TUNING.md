# Validation & Iterative Tuning Guide (`docs/VALIDATION_AND_TUNING.md`)

## 1. Phase 5 Validation Suite (`validate.py`)

Run `python3 validate.py` to execute all five automated sanity and calibration checks:

1. **Null Model Shuffle Test (5.1)**: Randomly permutes `bucket_id` labels across steps within each session over 10 Monte Carlo trials. Confirms that the positive slope of rot score vs. context fill collapses to flat (`<25%` of original slope), ruling out bucket-index leakage.
2. **Spot-Check Transcript Audit (5.1)**: Selects 3 high-rot buckets (`score_raw > 700`) and 3 pristine buckets (`score_raw < 100`), saves their transcripts and firing signals to `outputs/spot_check_transcripts.json`, and verifies that human-verifiable failure criteria disagree $\le 30\%$ of the time.
3. **Determinism & Stability Test (5.1)**: Runs `signals.py` $\rightarrow$ `score.py` $\rightarrow` `analyze.py` twice on identical inputs and verifies identical SHA-256 digests and zero floating-point drift.
4. **External Reference Precision@5 (5.2)**: Verifies that at least 80% of the top-5 highest-scoring buckets originate from known off-rails sessions.
5. **Cold-Start Bootstrap Check (5.3)**: Verifies that running with `--bootstrap` on a single session produces bounded, realistic scores using `bootstrap_priors` from `config.yaml`.

## 2. Phase 6 Iteration Loop (Every ~20 Sessions)

1. **Re-run the pipeline**:
   ```bash
   python3 run_pipeline.py --db-path ~/.local/share/opencode/opencode.db
   ```
2. **Check threshold stability**: Compare the recommended compaction threshold `T` in `outputs/report.html` against your previous run. It should stabilize within $\pm 20\text{k}$ tokens once you have $\ge 10$ sessions.
3. **Prune dead or saturated signals**:
   - Inspect `outputs/signal_rates.csv`.
   - If a signal's `peak_rate` is `0.000` across 20+ sessions, either refine its extractor in `signals.py` or set its weight to `0.0` in `config.yaml`.
   - If a signal fires on $>50\%$ of early steps ($\le 20\text{k}$ tokens), it is measuring a base habit rather than context rot — tighten its threshold or lower its weight.
