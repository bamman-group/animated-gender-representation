"""Extracts any archives in data/raw/ (including the official recognition
evaluation-code bundle) and verifies the official iCartoonFace release layout
is present. No reorganization is needed: rectrain images are already one
identity per subdirectory (torchvision ImageFolder-compatible), and the
rectest identification split (distractors vs. probe identities) is read
directly from icartoonface_rectest_info.txt by src/evaluate.py.

Expected layout under data/raw/ after extraction:
    personai_icartoonface_rectrain/icartoonface_rectrain/<identity_dir>/<image>.jpg
    personai_icartoonface_rectrain/icartoonface_rectrain_det.txt
    personai_icartoonface_rectest/icartoonface_rectest/<image>.jpg
    icartoonface_rectest_info.txt

The Google Drive dataset folder's copy of icartoonface_rectest_info.txt only
has the per-image bbox/label rows (22,500 lines). The official recognition
evaluation-code bundle (downloaded separately by scripts/download_data.sh,
see icartoonface_rec_evaluation_code.zip) ships a complete version of the
same file that also includes the explicit probe/gallery pair list (~209k
lines total). This script picks whichever copy has the most lines - after
extraction there may be more than one on disk - and copies it to the
canonical data/raw/icartoonface_rectest_info.txt path that
src/evaluate.py/src/evaluate_baseline.py expect.
"""
import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path

RAW_DIR = Path("data/raw")

TRAIN_IMAGE_DIR = "personai_icartoonface_rectrain/icartoonface_rectrain"
TRAIN_DET_FILE = "personai_icartoonface_rectrain/icartoonface_rectrain_det.txt"
TEST_IMAGE_DIR = "personai_icartoonface_rectest/icartoonface_rectest"
TEST_INFO_FILE = "icartoonface_rectest_info.txt"


def extract_archives(root: Path) -> None:
    for path in list(root.rglob("*")):
        if path.suffix == ".zip":
            print(f"Extracting {path} ...")
            with zipfile.ZipFile(path) as zf:
                zf.extractall(path.parent)
        elif path.suffix in (".tar", ".gz", ".bz2") and tarfile.is_tarfile(path):
            print(f"Extracting {path} ...")
            with tarfile.open(path) as tf:
                tf.extractall(path.parent)


def install_most_complete_test_info_file(root: Path, canonical_path: Path) -> None:
    """Finds every icartoonface_rectest_info.txt under root and copies the one
    with the most lines (i.e. the version that also has the pair list) to
    canonical_path, if that isn't already the most complete copy."""
    candidates = [p for p in root.rglob(TEST_INFO_FILE) if p.is_file()]
    if not candidates:
        return

    def line_count(path: Path) -> int:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)

    best = max(candidates, key=line_count)
    if best.resolve() == canonical_path.resolve():
        return

    print(f"Using {best} ({line_count(best)} lines) as the complete test info file.")
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, canonical_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    if not args.raw_dir.exists():
        raise SystemExit(f"{args.raw_dir} does not exist. Run scripts/download_data.sh first.")

    extract_archives(args.raw_dir)
    install_most_complete_test_info_file(args.raw_dir, args.raw_dir / TEST_INFO_FILE)

    required = {
        "train images": args.raw_dir / TRAIN_IMAGE_DIR,
        "train bbox file": args.raw_dir / TRAIN_DET_FILE,
        "test images": args.raw_dir / TEST_IMAGE_DIR,
        "test info/label file": args.raw_dir / TEST_INFO_FILE,
    }
    missing = {name: path for name, path in required.items() if not path.exists()}
    if missing:
        lines = "\n".join(f"  - {name}: expected at {path}" for name, path in missing.items())
        raise SystemExit(
            "Missing expected files/directories after extraction:\n"
            f"{lines}\n"
            f"Inspect {args.raw_dir} and update the *_DIR / *_FILE constants at the top "
            "of this script if the release layout has changed."
        )

    num_identities = sum(1 for p in (args.raw_dir / TRAIN_IMAGE_DIR).iterdir() if p.is_dir())
    num_train_images = sum(1 for _ in (args.raw_dir / TRAIN_IMAGE_DIR).rglob("*.jpg"))
    num_test_images = sum(1 for _ in (args.raw_dir / TEST_IMAGE_DIR).glob("*.jpg"))
    print(f"OK: {num_identities} training identities, {num_train_images} training images.")
    print(f"OK: {num_test_images} test images referenced by {TEST_INFO_FILE}.")
    print("No reorganization needed - these are the default paths the eval/train scripts expect.")


if __name__ == "__main__":
    main()
