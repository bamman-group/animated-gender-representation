#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path

RECOG_THRESHOLD_BY_CORPUS = {"animated": 0.42, "live_action": 0.18}


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
        try:
            name, score = ranked.split()[0].rsplit(":", 1)
            if float(score) >= threshold:
                track_name[track] = name
        except Exception as e:
            print(path, e)
    return track_name


def read_track_frames(path):
    track_frames = defaultdict(set)
    for line in path.read_text().splitlines():
        track, frame = line.split("\t")[:2]
        track_frames[track].add(frame)
    return track_frames


def load_imdb_ids(path, id_col, imdb_col):
    with open(path, newline="") as f:
        return {row[id_col]: row[imdb_col] for row in csv.DictReader(f, delimiter="\t")}


def movie_paths(stem, data_dir):
    return (
        data_dir / "fps" / f"{stem}.fps.txt",
        data_dir / "recog" / f"{stem}.recog.txt",
        data_dir / "tracks" / f"{stem}.tracks.txt",
    )


def find_stems(data_dir):
    return sorted(p.stem[: -len(".recog")] for p in (data_dir / "recog").glob("*.recog.txt"))


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
    script_dir = Path(__file__).resolve().parent
    analysis_data_dir = script_dir.parent / "data"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--corpus", choices=["animated", "live_action"], default="animated")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--metadata-id-col", default=None)
    parser.add_argument("--metadata-imdb-col", default=None)
    parser.add_argument("--recog-threshold", type=float, default=None)
    args = parser.parse_args()

    if args.corpus == "live_action":
        metadata_path = args.metadata or analysis_data_dir / "liveaction_metadata.tsv"
        metadata_id_col = args.metadata_id_col or "ID"
        metadata_imdb_col = args.metadata_imdb_col or "imdb"
    else:
        metadata_path = args.metadata or analysis_data_dir / "animation_metadata.tsv"
        metadata_id_col = args.metadata_id_col or "movie_id"
        metadata_imdb_col = args.metadata_imdb_col or "imdb_id"

    threshold = args.recog_threshold if args.recog_threshold is not None else RECOG_THRESHOLD_BY_CORPUS[args.corpus]
    imdb_by_stem = load_imdb_ids(metadata_path, metadata_id_col, metadata_imdb_col)
    stems = find_stems(args.data_dir)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if args.corpus == "live_action":
            writer.writerow(["movie_id", "movie_imdb", "actor_imdb", "seconds"])
            for stem in stems:
                movie_imdb = imdb_by_stem.get(stem, "")
                try:
                    for actor_imdb, seconds in screentime_for_movie(stem, args.data_dir, threshold).items():
                        writer.writerow([stem, movie_imdb, actor_imdb, f"{seconds:.2f}"])
                except Exception as e:
                    print(stem, movie_imdb, e)
        else:
            writer.writerow(["movie_id", "movie_imdb", "character", "seconds"])
            for stem in stems:
                movie_imdb = imdb_by_stem.get(stem, "")
                try:
                    for character, seconds in screentime_for_movie(stem, args.data_dir, threshold).items():
                        writer.writerow([stem, movie_imdb, character, f"{seconds:.2f}"])
                except Exception as e:
                    print(stem, movie_imdb, e)

if __name__ == "__main__":
    main()
