#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

RECOG_THRESHOLD_BY_CORPUS = {"animated": 0.42, "live_action": 0.18}


def normalize(name):
    return name.replace(" ", "_")


def read_fps(path):
    fields = path.read_text().strip().split("\t")
    return float(fields[4])


def read_recog(path, threshold):
    track_name = {}
    for line in path.read_text().splitlines():
        fields = line.split("\t")
        track, ranked = fields[0], fields[3] if len(fields) > 3 else ""
        if not ranked.strip():
            continue
        name, score = ranked.split()[0].rsplit(":", 1)
        if float(score) >= threshold:
            track_name[track] = name
    return track_name


def read_track_frames(path):
    track_frames = defaultdict(set)
    for line in path.read_text().splitlines():
        track, frame = line.split("\t")[:2]
        track_frames[track].add(frame)
    return track_frames


def load_years(path, id_col, year_col):
    with open(path, newline="") as f:
        return {row[id_col]: row[year_col] for row in csv.DictReader(f, delimiter="\t")}


def load_imdb_ids(path, id_col, imdb_col):
    with open(path, newline="") as f:
        return {row[id_col]: row[imdb_col] for row in csv.DictReader(f, delimiter="\t")}


def load_allowed_ids(path, id_col, filter_col, filter_value):
    if not filter_col:
        return None
    with open(path, newline="") as f:
        return {
            row[id_col] for row in csv.DictReader(f, delimiter="\t") if row[filter_col] == filter_value
        }


def load_actor_gender_history(path):
    history = {}
    with open(path, newline="") as f:
        for line in f:
            nm_id, name, gender_json = line.rstrip("\n").split("\t")
            years = {int(year): gender for year, gender in json.loads(gender_json).items()}
            history[nm_id] = (name, years)
    return history


def get_gender_for_year(gender_json, year):
    ordered_years = sorted(gender_json.keys())
    if len(ordered_years) == 1:
        return gender_json[ordered_years[0]]
    for idx, thisyear in enumerate(ordered_years):
        if idx == len(ordered_years) - 1:
            return gender_json[thisyear]
        nextyear = ordered_years[idx + 1]
        if year >= thisyear and year < nextyear:
            return gender_json[thisyear]


def load_roles(path):
    roles = defaultdict(dict)
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            roles[row["movie_id"]][row["actor_imdb"]] = (
                row["character_name"],
                row["role"],
                row["importance"],
            )
    return roles


def load_character_info(path):
    info = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            key = (row["movie id"], normalize(row["character_name"]))
            info[key] = (
                row["gender"],
                row.get("wikipedia role", ""),
                row.get("category", ""),
                row.get("actor IMDB", ""),
            )
    return info


def find_stems(data_dir):
    return sorted(p.stem[: -len(".recog")] for p in (data_dir / "recog").glob("*.recog.txt"))


def movie_paths(stem, data_dir):
    return (
        data_dir / "fps" / f"{stem}.fps.txt",
        data_dir / "recog" / f"{stem}.recog.txt",
        data_dir / "tracks" / f"{stem}.tracks.txt",
    )


def screentime_for_movie(stem, data_dir, threshold):
    fps_path, recog_path, tracks_path = movie_paths(stem, data_dir)
    fps = read_fps(fps_path)
    track_name = read_recog(recog_path, threshold)
    track_frames = read_track_frames(tracks_path)

    name_frames = defaultdict(set)
    for track, name in track_name.items():
        name_frames[name].update(track_frames.get(track, ()))

    return {name: len(frames) / fps for name, frames in name_frames.items()}


def main():
    analysis_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=analysis_dir / "data")
    parser.add_argument("--corpus", choices=["animated", "live_action"], default="animated")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--metadata-id-col", default=None)
    parser.add_argument("--metadata-year-col", default=None)
    parser.add_argument("--metadata-imdb-col", default=None)
    parser.add_argument("--metadata-filter-col", default=None)
    parser.add_argument("--metadata-filter-value", default=None)
    parser.add_argument("--gender-file", type=Path, default=None)
    parser.add_argument("--actor-gender-history", type=Path, default=None)
    parser.add_argument("--antagonist-file", type=Path, default=None)
    parser.add_argument("--recog-threshold", type=float, default=None)
    args = parser.parse_args()

    threshold = args.recog_threshold if args.recog_threshold is not None else RECOG_THRESHOLD_BY_CORPUS[args.corpus]

    if args.corpus == "live_action":
        metadata_path = args.metadata or analysis_dir / "data" / "liveaction_metadata.tsv"
        metadata_id_col = args.metadata_id_col or "ID"
        metadata_year_col = args.metadata_year_col or "year"
        metadata_imdb_col = args.metadata_imdb_col or "imdb"
        output = args.output or analysis_dir / "data" / "live_action_screentime_by_role.tsv"
    else:
        metadata_path = args.metadata or analysis_dir / "data" / "animation_metadata.tsv"
        metadata_id_col = args.metadata_id_col or "movie_id"
        metadata_year_col = args.metadata_year_col or "imdb_date"
        output = args.output or analysis_dir / "data" / "animated_screentime_by_role.tsv"

    years = load_years(metadata_path, metadata_id_col, metadata_year_col)
    allowed_ids = load_allowed_ids(
        metadata_path, metadata_id_col, args.metadata_filter_col, args.metadata_filter_value
    )
    stems = find_stems(args.data_dir)
    if allowed_ids is not None:
        stems = [s for s in stems if s in allowed_ids]

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "movie_id", "year", "id", "character_name", "actor_name", "actor_imdb",
                "gender", "role", "importance", "seconds",
            ]
        )

        if args.corpus == "live_action":
            imdb_by_stem = load_imdb_ids(metadata_path, metadata_id_col, metadata_imdb_col)
            actor_gender_history_path = (
                args.actor_gender_history or analysis_dir / "data" / "wikidata.actor.historical.gender.080526.tsv"
            )
            actor_gender_history = load_actor_gender_history(actor_gender_history_path)
            antagonist_file = args.antagonist_file or analysis_dir / "data" / "live_action_protagonist_antagonist.tsv"
            roles = load_roles(antagonist_file)

            for stem in stems:
                year_str = years.get(stem, "")
                year = int(float(year_str)) if year_str else None
                movie_roles = roles.get(imdb_by_stem.get(stem, ""), {})

                for nm_id, seconds in screentime_for_movie(stem, args.data_dir, threshold).items():
                    actor_name, gender_json = actor_gender_history.get(nm_id, ("", {}))
                    gender = (
                        get_gender_for_year(gender_json, year)
                        if gender_json and year is not None
                        else ""
                    )
                    character_name, role, importance = movie_roles.get(nm_id, ("", "", ""))
                    writer.writerow(
                        [
                            stem, year_str, nm_id, character_name, actor_name, nm_id,
                            gender, role, importance, f"{seconds:.2f}",
                        ]
                    )
        else:
            gender_file = args.gender_file or analysis_dir / "data" / "character_gender.tsv"
            char_info = load_character_info(gender_file)

            for stem in stems:
                year_str = years.get(stem, "")

                for character, seconds in screentime_for_movie(stem, args.data_dir, threshold).items():
                    gender, role, _category, actor_imdb = char_info.get(
                        (stem, character), ("", "", "", "")
                    )
                    writer.writerow(
                        [
                            stem, year_str, character, character, "", actor_imdb,
                            gender, role, "", f"{seconds:.2f}",
                        ]
                    )

    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
