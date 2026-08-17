#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

ROLES = ["protagonist", "antagonist"]
GENDERS = ["female", "male"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    years = {}
    # {movie_id: {gender: {role: seconds}}}
    totals = {}

    with open(args.input, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            role = row["role"]
            if role not in ROLES:
                continue
            gender_labels = row["gender"].split("#") if row["gender"] else []
            matched_genders = [g for g in GENDERS if g in gender_labels]
            if not matched_genders:
                continue

            movie_id = row["movie_id"]
            years[movie_id] = row["year"]
            t = totals.setdefault(movie_id, {g: {r: 0.0 for r in ROLES} for g in GENDERS})
            seconds = float(row["seconds"])
            for gender in matched_genders:
                t[gender][role] += seconds

    def pct(numerator, denominator):
        return f"{100 * numerator / denominator:.2f}" if denominator else ""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "movie_id", "year",
                "pct_protagonist_female", "pct_antagonist_female",
                "pct_protagonist_male", "pct_antagonist_male",
            ]
        )
        for movie_id, year in years.items():
            t = totals[movie_id]
            female_denom = t["female"]["protagonist"] + t["female"]["antagonist"]
            male_denom = t["male"]["protagonist"] + t["male"]["antagonist"]
            writer.writerow(
                [
                    movie_id, year,
                    pct(t["female"]["protagonist"], female_denom),
                    pct(t["female"]["antagonist"], female_denom),
                    pct(t["male"]["protagonist"], male_denom),
                    pct(t["male"]["antagonist"], male_denom),
                ]
            )


if __name__ == "__main__":
    main()
