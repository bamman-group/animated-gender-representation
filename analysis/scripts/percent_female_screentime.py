#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def normalize(character):
    return character.replace(" ", "_")


def load_screentime(path):
    screentime = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            screentime[(row["movie_id"], normalize(row["character"]))] = float(row["seconds"])
    return screentime


def new_totals():
    return {"all": [0.0, 0.0], "human": [0.0, 0.0], "animal_other": [0.0, 0.0]}


def compute_totals(gender_file, screentime):
    totals = {}
    with open(gender_file, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["gender"] == "unknown":
                continue
            key = (row["movie id"], normalize(row["character_name"]))
            seconds = screentime.get(key, 0.0)
            is_female = row["gender"] == "female"

            buckets = ["all"]
            if row["category"] == "human":
                buckets.append("human")
            elif row["category"] in ("animal", "other"):
                buckets.append("animal_other")

            t = totals.setdefault(row["movie id"], new_totals())
            for bucket in buckets:
                t[bucket][1] += seconds
                if is_female:
                    t[bucket][0] += seconds
    return totals


def pct(num, denom):
    return f"{100 * num / denom:.2f}" if denom else ""


def main():
    analysis_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--screentime", type=Path, default=analysis_dir / "data" / "animated_screentime.tsv")
    parser.add_argument("--gender-file", type=Path, default=analysis_dir / "data" / "character_gender.tsv")
    parser.add_argument("--metadata", type=Path, default=analysis_dir / "data" / "animation_metadata.tsv")
    parser.add_argument(
        "--output", type=Path, default=analysis_dir / "data" / "animated_screentime_gender.tsv"
    )
    args = parser.parse_args()

    screentime = load_screentime(args.screentime)
    totals = compute_totals(args.gender_file, screentime)

    new_columns = [
        "% screentime, female (all)",
        "% screentime, female (human)",
        "% screentime, female (animal/other)",
    ]
    with open(args.metadata, newline="") as f_in, open(args.output, "w", newline="") as f_out:
        reader = csv.DictReader(f_in, delimiter="\t")
        writer = csv.writer(f_out, delimiter="\t")
        writer.writerow(reader.fieldnames + new_columns)
        for row in reader:
            t = totals.get(row["movie_id"], new_totals())
            writer.writerow(
                list(row.values()) + [pct(*t["all"]), pct(*t["human"]), pct(*t["animal_other"])]
            )

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
