#!/usr/bin/env python
import csv
import json
import sys
import statistics as stats
from collections import Counter
from pathlib import Path

def load_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def parse_inputs_email_name(row):
    try:
        inp = json.loads(row.get("inputs") or "{}")
        return inp.get("email_name") or inp.get("inputs", {}).get("email_name")
    except Exception:
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_results_csv.py <path-to-studio-export.csv>")
        sys.exit(2)
    path = Path(sys.argv[1]).resolve()
    rows = load_rows(path)
    total = len(rows)
    status_counts = Counter((r.get("status") or "").lower() or "unknown" for r in rows)
    lat_vals = []
    for r in rows:
        v = r.get("latency")
        if v:
            try:
                lat_vals.append(float(v))
            except Exception:
                pass

    fails = [r for r in rows if (r.get("status") or "").lower() != "success"]
    fail_by_name = Counter()
    for r in fails:
        name = parse_inputs_email_name(r)
        if name:
            fail_by_name[name] += 1

    print("Summary for:", path)
    print("- Total:", total)
    print("- Status counts:", dict(status_counts))
    if lat_vals:
        print(
            "- Latency ms (avg/min/max):",
            round(stats.mean(lat_vals), 1),
            "/",
            round(min(lat_vals), 1),
            "/",
            round(max(lat_vals), 1),
        )
    print("- Failing cases:", len(fails))
    if fail_by_name:
        print("- Top failing email_name:")
        for name, cnt in fail_by_name.most_common(10):
            print(f"  * {name}: {cnt}")

    # Show one example fail details (trimmed)
    if fails:
        r = fails[0]
        print("\nExample failing row id:", r.get("id"))
        err = r.get("error")
        if err:
            print("- error:", (err[:300] + "...") if len(err) > 300 else err)
        try:
            outputs = json.loads(r.get("outputs") or "{}")
            msgs = outputs.get("messages") or []
            print("- last message keys:", list((msgs[-1] or {}).keys()) if msgs else [])
        except Exception:
            pass

if __name__ == "__main__":
    main()

