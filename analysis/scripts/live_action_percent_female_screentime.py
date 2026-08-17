#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_gender_history(path):
    history = {}
    with open(path, newline="", encoding="utf-8") as f:
        for line in f:
            nm_id, _name, gender_json = line.rstrip("\n").split("\t")
            years = {int(year): gender for year, gender in json.loads(gender_json).items()}
            history[nm_id] = years
    return history


def gender_for_year(years, year):
    # latest recorded year <= target year, or the earliest entry if target
    # year precedes every recorded year
    ordered_years = sorted(years)
    if len(ordered_years) == 1:
        return years[ordered_years[0]]
    for idx, this_year in enumerate(ordered_years):
        if idx == len(ordered_years) - 1:
            return years[this_year]
        next_year = ordered_years[idx + 1]
        if this_year <= year < next_year:
            return years[this_year]


def load_screentime(path):
    by_movie = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            by_movie[row["movie_id"]].append((row["actor_imdb"], float(row["seconds"])))
    return by_movie


def gender_percentages(rows, gender_history, year):
    female_seconds = 0.0
    male_seconds = 0.0
    known_seconds = 0.0

    for actor_imdb, seconds in rows:
        years = gender_history.get(actor_imdb)
        if not years or year is None:
            continue
        gender = gender_for_year(years, year)
        if not gender:
            continue
        known_seconds += seconds
        labels = gender.split("#")
        if "female" in labels:
            female_seconds += seconds
        if "male" in labels:
            male_seconds += seconds

    if known_seconds == 0:
        return "", ""
    return f"{100 * female_seconds / known_seconds:.4f}", f"{100 * male_seconds / known_seconds:.4f}"


def main():
    analysis_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gender-history", type=Path,
        default=analysis_dir / "data" / "wikidata.actor.historical.gender.080526.tsv",
    )
    parser.add_argument("--screentime", type=Path, default=analysis_dir / "data" / "liveaction_screentime.tsv")
    parser.add_argument("--metadata", type=Path, default=analysis_dir / "data" / "liveaction_metadata.tsv")
    parser.add_argument("--metadata-id-col", default="ID")
    parser.add_argument("--metadata-year-col", default="year")
    parser.add_argument(
        "--output", type=Path, default=analysis_dir / "data" / "liveaction_screentime_gender.tsv"
    )
    args = parser.parse_args()

    gender_history = load_gender_history(args.gender_history)
    screentime_by_movie = load_screentime(args.screentime)

    with open(args.metadata, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames + ["% screentime, female", "% screentime, male"]
        rows = list(reader)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            movie_id = row[args.metadata_id_col]
            year_str = row.get(args.metadata_year_col, "")
            year = int(float(year_str)) if year_str else None
            movie_rows = screentime_by_movie.get(movie_id, [])
            pct_female, pct_male = gender_percentages(movie_rows, gender_history, year)
            row["% screentime, female"] = pct_female
            row["% screentime, male"] = pct_male
            writer.writerow(row)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
