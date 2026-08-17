#!/usr/bin/env python3
"""
RetailCast India - pre-submission FORMAT validator (participant-safe; no answer key).

Checks that your submission.csv is structurally valid BEFORE you submit. It does NOT score
accuracy (only the organizers can, against the held-out horizon). It mirrors the structural
checks the evaluation agent runs, so if this passes, your file will be accepted for scoring.

Usage:
  python3 validate_format.py --submission path/to/submission.csv [--sample sample_submission.csv]
"""
import argparse, os, sys
import pandas as pd

FCOLS = [f"F{i}" for i in range(1, 29)]

def main(sub_path, sample_path):
    if not os.path.exists(sub_path):
        sys.exit(f"FAIL: file not found: {sub_path}")
    ids = pd.read_csv(sample_path, dtype=str)["id"].str.strip().tolist()
    try:
        df = pd.read_csv(sub_path, dtype=str, keep_default_na=False, skipinitialspace=True)
    except Exception as e:
        sys.exit(f"FAIL: could not parse CSV: {e}")
    df = df.rename(columns={c: c.strip() for c in df.columns})
    df = df.rename(columns={c: ("id" if c.lower() == "id" else c.upper() if c.upper() in FCOLS else c)
                            for c in df.columns})
    problems = []
    if "id" not in df.columns: problems.append("no 'id' column")
    missing = [c for c in FCOLS if c not in df.columns]
    if missing: problems.append(f"missing forecast columns: {missing[:6]}{'...' if len(missing)>6 else ''}")
    if problems:
        for p in problems: print("FAIL:", p)
        sys.exit(1)
    if len(df) != 60: problems.append(f"expected 60 rows, found {len(df)}")
    sid = df["id"].str.strip()
    dup = sid[sid.duplicated()].unique().tolist()
    if dup: problems.append(f"duplicate ids: {dup[:3]}")
    unknown = sorted(set(sid) - set(ids)); miss = sorted(set(ids) - set(sid))
    if unknown: problems.append(f"unknown ids (not in sample): {unknown[:3]}")
    if miss: problems.append(f"missing ids: {miss[:3]}")
    vals = df[FCOLS].apply(lambda s: pd.to_numeric(s, errors="coerce"))
    n_bad = int(vals.isna().sum().sum())
    if n_bad: problems.append(f"{n_bad} non-numeric/empty forecast cell(s)")
    else:
        arr = vals.to_numpy(float)
        if (arr < 0).any(): problems.append(f"{int((arr<0).sum())} negative forecast value(s)")
        import numpy as np
        if not np.isfinite(arr).all(): problems.append("inf/NaN present")
    if problems:
        for p in problems: print("FAIL:", p)
        sys.exit(1)
    print("PASS: submission.csv is structurally valid (60 rows, id + F1..F28, numeric, non-negative).")
    print("Note: this checks FORMAT only. Accuracy is scored by the organizers against held-out data.")

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--sample", default=os.path.join(here, "sample_submission.csv"))
    a = ap.parse_args()
    main(a.submission, a.sample)
