import csv, json, os, re, sys

root, rc, started = sys.argv[1], int(sys.argv[2]), sys.argv[3]
log_path = os.path.join(root, "history", "last_run.log")
curve_path = os.path.join(root, "outputs", "score_curve.csv")
val_path = os.path.join(root, "outputs", "validation_results.json")

sessions = steps = max_since = rec = knee = ""
scores = {}

try:
    log = open(log_path, errors="ignore").read()
    m = re.search(r"(\d+)\s+steps across\s+(\d+)\s+sessions", log)
    if m:
        steps, sessions = m.group(1), m.group(2)
    m = re.search(r"max tokens_since_compaction=([\d,]+)", log)
    if m:
        max_since = m.group(1).replace(",", "")
except Exception:
    pass

try:
    with open(curve_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        try:
            scores[int(float(r["tokens_since_compaction_k"]))] = float(r["mean_score"])
        except Exception:
            continue
except Exception:
    pass


def at(target_k):
    """Score at the closest bucket at or below target_k — the curve is 5k-bucketed, so an
    exact key usually does not exist and a missing column would silently hide the trend."""
    keys = [k for k in scores if k <= target_k]
    if not keys:
        return ""
    return f"{scores[max(keys)]:.2f}"


try:
    v = json.load(open(val_path))
    for c in v.get("checks", []):
        d = c.get("details", "")
        m = re.search(r"recommended T = ([\d,]+)", d)
        if m:
            rec = m.group(1).replace(",", "")
        m = re.search(r"Knee threshold T_\w+=([\d,]+)", d)
        if m:
            knee = m.group(1).replace(",", "")
except Exception:
    pass

status = "ok" if rc == 0 else f"rc{rc}"
print(",".join([started, status, str(sessions), str(steps), str(max_since), str(rec),
                str(knee), at(64), at(128), at(256)]))
