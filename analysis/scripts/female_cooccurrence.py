#!/usr/bin/env python3
import argparse
import bisect
import csv
import json
from collections import defaultdict
from pathlib import Path

RECOG_THRESHOLD_BY_CORPUS = {"animated": 0.42, "live_action": 0.18}


def normalize(character):
    return character.replace(" ", "_")


def read_shots(path):
    shots = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        start, end = line.split()[:2]
        shots.append((int(start), int(end)))
    return shots


def shot_lookup(shots):
    starts = [s[0] for s in shots]

    def lookup(frame):
        idx = bisect.bisect_right(starts, frame) - 1
        if idx < 0:
            return None
        start, end = shots[idx]
        return idx if start <= frame <= end else None

    return lookup


def read_recog(path, threshold):
    track_character = {}
    for line in path.read_text().splitlines():
        fields = line.split("\t")
        track, ranked = fields[0], fields[3] if len(fields) > 3 else ""
        if not ranked.strip():
            continue
        name, score = ranked.split()[0].rsplit(":", 1)
        if float(score) >= threshold:
            track_character[track] = name
    return track_character


def read_track_frames(path):
    track_frames = defaultdict(set)
    for line in path.read_text().splitlines():
        track, frame = line.split("\t")[:2]
        track_frames[track].add(int(frame))
    return track_frames


def load_female_characters(path):
    female = defaultdict(set)
    female_human = defaultdict(set)
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["gender"] != "female":
                continue
            name = normalize(row["character_name"])
            female[row["movie id"]].add(name)
            if row.get("category", "human") == "human":
                female_human[row["movie id"]].add(name)
    return female, female_human


def load_years(path, id_col, year_col):
    with open(path, newline="") as f:
        return {row[id_col]: row[year_col] for row in csv.DictReader(f, delimiter="\t")}


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
            nm_id, _name, gender_json = line.rstrip("\n").split("\t")
            history[nm_id] = {int(year): gender for year, gender in json.loads(gender_json).items()}
    return history


def get_gender_for_year(gender_json, year):
    ordered_years = sorted(gender_json.keys())
    if len(ordered_years) == 1:
        return gender_json[ordered_years[0]].split("#")
    for idx, thisyear in enumerate(ordered_years):
        if idx == len(ordered_years) - 1:
            return gender_json[thisyear].split("#")
        nextyear = ordered_years[idx + 1]
        if year >= thisyear and year < nextyear:
            return gender_json[thisyear].split("#")


def is_female_actor_at_year(actor_gender_history, nm_id, year):
    gender_json = actor_gender_history.get(nm_id)
    if not gender_json or year is None:
        return False
    return "female" in get_gender_for_year(gender_json, year)


def shot_to_target_characters(track_character, track_frames, target_characters, frame_to_shot):
    shot_targets = defaultdict(set)
    for track, character in track_character.items():
        if character not in target_characters:
            continue
        for frame in track_frames.get(track, ()):
            shot_idx = frame_to_shot(frame)
            if shot_idx is None:
                continue
            shot_targets[shot_idx].add(character)
    return shot_targets


def two_plus_and_one_plus_counts(total_shots, shot_targets):
    # s/s+1 union window smooths over single-shot detection gaps
    shots_with_two_plus = 0
    shots_with_one_plus = 0
    for s in range(total_shots - 1):
        present = shot_targets.get(s, set()) | shot_targets.get(s + 1, set())
        n = len(present)
        if n >= 1:
            shots_with_one_plus += 1
        if n >= 2:
            shots_with_two_plus += 1
    return shots_with_two_plus, shots_with_one_plus


def ratio(numerator, denominator):
    return numerator / denominator if denominator else ""


def movie_paths(stem, data_dir):
    return (
        data_dir / "shots" / f"{stem}.scenes.txt",
        data_dir / "recog" / f"{stem}.recog.txt",
        data_dir / "tracks" / f"{stem}.tracks.txt",
    )


def find_stems(data_dir):
    return sorted(p.stem[: -len(".recog")] for p in (data_dir / "recog").glob("*.recog.txt"))


def cooccurrence_for_movie(stem, data_dir, is_female, is_human, threshold):
    shots_path, recog_path, tracks_path = movie_paths(stem, data_dir)
    shots = read_shots(shots_path)
    total_shots = len(shots)
    frame_to_shot = shot_lookup(shots)
    track_character = read_recog(recog_path, threshold)
    track_frames = read_track_frames(tracks_path)

    all_characters = set(track_character.values())
    female_characters = {n for n in all_characters if is_female(n)}
    female_human_characters = {n for n in female_characters if is_human(n)}

    shot_females = shot_to_target_characters(
        track_character, track_frames, female_characters, frame_to_shot
    )
    shot_human_females = shot_to_target_characters(
        track_character, track_frames, female_human_characters, frame_to_shot
    )

    two_plus_fem, one_plus_fem = two_plus_and_one_plus_counts(total_shots, shot_females)
    two_plus_human_fem, one_plus_human_fem = two_plus_and_one_plus_counts(
        total_shots, shot_human_females
    )

    ratio_all = ratio(two_plus_fem, total_shots)
    ratio_fem = ratio(two_plus_fem, one_plus_fem)
    ratio_all_human = ratio(two_plus_human_fem, total_shots)
    ratio_fem_human = ratio(two_plus_human_fem, one_plus_human_fem)

    return ratio_all, ratio_fem, ratio_all_human, ratio_fem_human


def main():
    analysis_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=analysis_dir / "data")
    parser.add_argument("--corpus", choices=["animated", "live_action"], default="animated")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--metadata-id-col", default=None)
    parser.add_argument("--metadata-year-col", default=None)
    parser.add_argument("--metadata-filter-col", default=None)
    parser.add_argument("--metadata-filter-value", default=None)
    parser.add_argument("--gender-file", type=Path, default=None)
    parser.add_argument("--actor-gender-history", type=Path, default=None)
    parser.add_argument("--recog-threshold", type=float, default=None)
    args = parser.parse_args()

    threshold = args.recog_threshold if args.recog_threshold is not None else RECOG_THRESHOLD_BY_CORPUS[args.corpus]

    if args.corpus == "live_action":
        metadata_path = args.metadata or analysis_dir / "data" / "liveaction_metadata.tsv"
        metadata_id_col = args.metadata_id_col or "ID"
        metadata_year_col = args.metadata_year_col or "year"
        output = args.output or analysis_dir / "data" / "live_action_cooccurrence.tsv"
    else:
        metadata_path = args.metadata or analysis_dir / "data" / "animation_metadata.tsv"
        metadata_id_col = args.metadata_id_col or "movie_id"
        metadata_year_col = args.metadata_year_col or "imdb_date"
        output = args.output or analysis_dir / "data" / "female_cooccurrence.tsv"

    years = load_years(metadata_path, metadata_id_col, metadata_year_col)
    allowed_ids = load_allowed_ids(
        metadata_path, metadata_id_col, args.metadata_filter_col, args.metadata_filter_value
    )
    stems = find_stems(args.data_dir)
    if allowed_ids is not None:
        stems = [s for s in stems if s in allowed_ids]

    if args.corpus == "live_action":
        actor_gender_history_path = (
            args.actor_gender_history or analysis_dir / "data" / "wikidata.actor.historical.gender.080526.tsv"
        )
        actor_gender_history = load_actor_gender_history(actor_gender_history_path)

        def make_tests(stem):
            year_str = years.get(stem, "")
            year = int(float(year_str)) if year_str else None
            is_female = lambda n: is_female_actor_at_year(actor_gender_history, n, year)
            is_human = lambda n: True
            return is_female, is_human
    else:
        gender_file = args.gender_file or analysis_dir / "data" / "character_gender.tsv"
        female_characters, female_human_characters = load_female_characters(gender_file)

        def make_tests(stem):
            is_female = lambda n: n in female_characters.get(stem, set())
            is_human = lambda n: n in female_human_characters.get(stem, set())
            return is_female, is_human

    def fmt(ratio):
        return f"{ratio:.6f}" if ratio != "" else ""

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            ["movie_id", "year", "ratio_all", "ratio_fem", "ratio_all_human", "ratio_fem_human"]
        )
        for stem in stems:
            is_female, is_human = make_tests(stem)
            ratio_all, ratio_fem, ratio_all_human, ratio_fem_human = cooccurrence_for_movie(
                stem, args.data_dir, is_female, is_human, threshold
            )
            writer.writerow(
                [
                    stem,
                    years.get(stem, ""),
                    fmt(ratio_all),
                    fmt(ratio_fem),
                    fmt(ratio_all_human),
                    fmt(ratio_fem_human),
                ]
            )

    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
