#!/usr/bin/env python3
import argparse
import csv
import random

import numpy as np

random.seed(1)
np.random.seed(1)

B = 10000


def read_data(filename, value_col, year_col, filter_col, filter_value):
    data = {}
    with open(filename, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if filter_col and row.get(filter_col) != filter_value:
                continue
            val_str = (row.get(value_col) or "").strip()
            year_str = (row.get(year_col) or "").strip()
            if not val_str or val_str in ("None", "NA") or not year_str:
                continue
            year = int(float(year_str))
            decade = (year // 10) * 10
            data.setdefault(decade, []).append(float(val_str))
    return data


def proc(data):
    for decade in sorted(data):
        vals = np.array(data[decade], dtype=float)
        means = [np.mean(np.random.choice(vals, size=len(vals), replace=True)) for _ in range(B)]
        p2_5, p50, p97_5 = np.percentile(means, [2.5, 50, 97.5])
        print("%d\t%.5f\t%.5f\t%.5f\t%d" % (decade, p2_5, p50, p97_5, len(vals)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("value_col")
    parser.add_argument("--year-col", default="year")
    parser.add_argument("--filter-col")
    parser.add_argument("--filter-value")
    args = parser.parse_args()

    data = read_data(args.filename, args.value_col, args.year_col, args.filter_col, args.filter_value)
    proc(data)


if __name__ == "__main__":
    main()
