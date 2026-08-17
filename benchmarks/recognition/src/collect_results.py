"""Generate the paper's LaTeX tables from the shared results.md and the
per-run outputs/*/timing.json files - the recognition-side counterpart of
the detection benchmark's src/eval/collect_results.py.

--table variants     (default): one table per evaluation set (iCartoonFace, Film), rows
                     buffalo_l / dino_vitb14 / dino_vitl14, columns Base /
                     FT / I+T/Train / I+T / im/s (baseline, fine-tuned
                     identity-only, identity+tracks trained on ONLY the
                     movies known to be safe (unannotated, no character
                     overlap with data/film) - see
                     scripts/slurm/submit_trainonly_tracks_jobs.sh - identity+tracks,
                     inference throughput). Each model gets two lines: crop 0%
                     on its own row, crop 25% indented just below. Accuracy
                     columns parsed from --results (default results.md) via
                     the "<model> (<variant>, crop N%)" --tag convention used
                     by scripts/evaluate_all.sh, submit_tracks_jobs.sh, and
                     submit_trainonly_tracks_jobs.sh. The im/s column is
                     parsed separately from --inference-timing (default
                     results/inference_timing.md, written by
                     src/benchmark_inference.py / scripts/benchmark_all.sh) -
                     one measurement per (model, crop), since inference speed
                     doesn't depend on which variant was fine-tuned; prefers a
                     fine-tuned-variant measurement at that crop, falling back
                     to the model's (crop-independent) baseline measurement if
                     none exists. Wrapped in \\resizebox{\\linewidth}{!}{...}
                     so it always fits one column regardless of content width
                     - requires \\usepackage{graphicx} in the including LaTeX
                     document.
--table training     training wall-clock seconds / epochs / throughput for
                     each fine-tuning run that wrote a timing.json.

Examples (from the repo root):
    python -m src.collect_results                        # -> results/table_variants.tex (default)
    python -m src.collect_results --table training      # -> results/table_training.tex
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Display order + short column header for the evaluation sets, matched by the
# dataset label the eval scripts log (see append_markdown_result callers).
EVAL_SETS = [
    ("iCartoonFace rectest", "iCartoonFace"),
    ("data/film (per-movie)", "Film"),
]

# Rows/columns for --table variants, matched against the "<model> (<variant>,
# crop N%)" --tag convention (scripts/evaluate_all.sh, submit_tracks_jobs.sh).
VARIANT_MODELS = ["buffalo_l", "dino_vitb14", "dino_vitl14"]
VARIANT_COLUMNS = [
    ("baseline", "Base"),
    ("fine-tuned", "FT"),
    ("identity+tracks/train", "I+T/Train"),
    ("identity+tracks", "I+T"),
]
VARIANT_CROPS = [0, 25]
_TAG_RE = re.compile(r"^(.*) \((.*), crop (\d+)%\)$")
# Looser than _TAG_RE: matches both fine-tuned rows ("<model> (<variant>,
# crop N%)") and baseline rows with no crop suffix at all ("<model>
# (baseline)"), as written by src/benchmark_inference.py.
_INFERENCE_TAG_RE = re.compile(r"^(.*) \(([^,()]+)(?:, crop (\d+)%)?\)$")


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%")


def parse_results(path: Path) -> dict[tuple[str, str], dict]:
    """results.md -> {(model, dataset): {'acc','ci','ims'}}, latest row wins.
    Row columns: Model | Dataset | Rank@1 | 95% CI | Correct/Total | im/s | Timestamp."""
    rows: dict[tuple[str, str], dict] = {}
    if not path.is_file():
        return rows
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6 or cells[0] in ("Model", ""):
            continue
        try:
            acc = float(cells[2])
        except ValueError:
            continue  # header separator / non-data line
        ims = cells[5]
        rows[(cells[0], cells[1])] = {
            "acc": acc,
            "ci": cells[3] if cells[3] and cells[3] != "-" else None,
            "ims": ims if ims and ims != "-" else None,
        }
    return rows


def parse_variant_results(rows: dict[tuple[str, str], dict]) -> dict[tuple[str, str, str, int], dict]:
    """Re-key parse_results()'s {(tag, dataset): ...} by
    (base_model, variant, dataset, crop_pct). tag must match the
    "<model> (<variant>, crop N%)" convention (scripts/evaluate_all.sh /
    submit_tracks_jobs.sh); rows that don't match this convention are skipped."""
    out: dict[tuple[str, str, str, int], dict] = {}
    for (tag, dataset), r in rows.items():
        m = _TAG_RE.match(tag)
        if not m:
            continue
        model, variant, crop_pct = m.group(1), m.group(2), int(m.group(3))
        out[(model, variant, dataset, crop_pct)] = r
    return out


def parse_inference_timing(path: Path) -> dict[tuple[str, str, int | None], float]:
    """results/inference_timing.md (src/benchmark_inference.py) ->
    {(model, variant, crop_pct): im/s}, latest row wins. Row columns: Model |
    Images | Batch Size | Seconds | im/s | Timestamp. crop_pct is None for
    baseline rows, which carry no crop suffix in their tag (inference speed
    is measured once for the baseline, not per crop)."""
    out: dict[tuple[str, str, int | None], float] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5 or cells[0] in ("Model", ""):
            continue
        m = _INFERENCE_TAG_RE.match(cells[0])
        if not m:
            continue
        model, variant, crop = m.group(1), m.group(2), m.group(3)
        try:
            ims = float(cells[4])
        except ValueError:
            continue  # header separator / non-data line
        out[(model, variant, int(crop) if crop is not None else None)] = ims
    return out


def variant_table(
    variant_rows: dict[tuple[str, str, str, int], dict],
    dataset_label: str,
    inference_rows: dict[tuple[str, str, int | None], float],
) -> str:
    def cell(model: str, variant: str, crop_pct: int) -> str:
        r = variant_rows.get((model, variant, dataset_label, crop_pct))
        if not r:
            return "--"
        if r["ci"]:
            return f"{r['acc']:.4f} {{\\tiny {_tex_escape(r['ci'])}}}"
        return f"{r['acc']:.4f}"

    def ims_cell(model: str, crop_pct: int) -> str:
        # Inference speed doesn't depend on which variant was fine-tuned, only
        # architecture + crop - prefer any fine-tuned-variant measurement at
        # this crop, falling back to the (crop-independent) baseline.
        for variant, _ in VARIANT_COLUMNS:
            if variant == "baseline":
                continue
            ims = inference_rows.get((model, variant, crop_pct))
            if ims is not None:
                return f"{ims:,.1f}"
        ims = inference_rows.get((model, "baseline", None))
        return f"{ims:,.1f}" if ims is not None else "--"

    col_spec = "l" + "c" * len(VARIANT_COLUMNS) + "r"
    headers = " & ".join(short for _, short in VARIANT_COLUMNS)
    lines = [
        "% generated by src/collect_results.py",
        "\\begin{table}",
        "\\centering",
        "\\resizebox{\\linewidth}{!}{%",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        f"Model & {headers} & im/s \\\\",
        "\\midrule",
    ]
    for model in VARIANT_MODELS:
        for i, crop_pct in enumerate(VARIANT_CROPS):
            cells = " & ".join(cell(model, variant, crop_pct) for variant, _ in VARIANT_COLUMNS)
            label = _tex_escape(model) if i == 0 else f"\\quad + {crop_pct}\\% padding"
            lines.append(f"{label} & {cells} & {ims_cell(model, crop_pct)} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "}",  # close \resizebox
        f"\\caption{{Rank@1 identification accuracy (95\\% bootstrap CI) on {_tex_escape(dataset_label)}, "
        "crop 0\\% and crop 25\\% (indented), plus inference throughput (images/second, from "
        "src/benchmark_inference.py). Base=baseline, FT=fine-tuned (identity only), "
        "I+T=identity+tracks, I+T/Train=identity+tracks trained on ONLY the movies known to be "
        "safe, with every other movie excluded (see --only-movies-file).}",
        f"\\label{{tab:recognition_variants_{'film' if 'film' in dataset_label.lower() else 'icartoonface'}}}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def variants_tables(
    rows: dict[tuple[str, str], dict], inference_rows: dict[tuple[str, str, int | None], float]
) -> str:
    variant_rows = parse_variant_results(rows)
    return "\n".join(variant_table(variant_rows, label, inference_rows) for label, _ in EVAL_SETS)


def training_table(runs_root: Path) -> str:
    timings = sorted(runs_root.glob("*/timing.json"))
    lines = [
        "% generated by src/collect_results.py",
        "\\begin{table}",
        "\\centering",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Model & Train (s) & Epochs & im/s \\\\",
        "\\midrule",
    ]
    for p in timings:
        t = json.loads(p.read_text())
        ims = t.get("images_per_sec")
        name = _tex_escape(str(t.get("model", p.parent.name)))
        ims_cell = f"{ims:,.1f}" if ims else "--"
        lines.append(
            f"{name} & {t.get('train_seconds', 0):,.0f} & {t.get('epochs_run', '--')} & {ims_cell} \\\\"
        )
    if not timings:
        lines.append("\\multicolumn{4}{c}{no timing.json found under %s} \\\\" % _tex_escape(str(runs_root)))
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Fine-tuning wall-clock time, epochs run, and training throughput "
        "(images/second), one row per run's \\texttt{timing.json}.}",
        "\\label{tab:recognition_training}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default="variants", choices=["variants", "training"])
    ap.add_argument("--results", default=str(REPO_ROOT / "results.md"))
    ap.add_argument("--inference-timing", default=str(REPO_ROOT / "results" / "inference_timing.md"),
                    help="[--table variants] src/benchmark_inference.py's output (im/s column)")
    ap.add_argument("--runs-root", default=str(REPO_ROOT / "outputs"),
                    help="[--table training] directory holding <run>/timing.json")
    ap.add_argument("--out", default=None, help="default: results/table_<table>.tex")
    args = ap.parse_args()

    if args.table == "training":
        table = training_table(Path(args.runs_root))
    else:
        table = variants_tables(parse_results(Path(args.results)), parse_inference_timing(Path(args.inference_timing)))

    out = Path(args.out or REPO_ROOT / "results" / f"table_{args.table}.tex")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table)
    print(table)
    print(f"% written to {out}")


if __name__ == "__main__":
    main()
