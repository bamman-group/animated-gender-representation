"""Shared helpers for writing evaluation/training results, used by
src/evaluate.py, src/evaluate_film.py, src/evaluate_baseline.py, and the
fine-tuning scripts (src/train_dino.py, src/train_buffalo.py) so that runs
against different datasets/models/checkpoints can be collected into one report
(pass the same --output/--timing-output path to each command you run).

Two formats:
  - Markdown (append_markdown_result): a human-readable table of Rank@1
    results, including its 95% bootstrap confidence interval and the
    inference throughput (images/sec) measured while embedding.
  - JSON Lines (append_json_record): one JSON object per line, for later
    programmatic analysis (e.g. `pandas.read_json(path, lines=True)`) of
    timing data - training epoch/total wall-clock time, inference
    images/sec, etc. src/collect_results.py turns both into LaTeX tables.
"""
import datetime
import json
from dataclasses import dataclass
from pathlib import Path

MARKDOWN_HEADER = (
    "| Model | Dataset | Rank@1 | 95% CI | Correct/Total | im/s | Timestamp |\n"
    "|---|---|---|---|---|---|---|\n"
)


@dataclass
class EvalResult:
    accuracy: float
    correct: int
    total: int
    embedding_seconds: float = 0.0
    embedding_count: int = 0
    # 95% bootstrap CI for accuracy (fractions in [0, 1]); None when the CI
    # was not computed (e.g. mid-training dev evals, which skip it for speed).
    ci_low: float | None = None
    ci_high: float | None = None
    # Mean reciprocal rank, computed only by the mid-training dev evals as a
    # smoother checkpoint-selection signal than binary Rank@1; None for the
    # final test protocols (which report Rank@1 only).
    mrr: float | None = None

    @property
    def images_per_sec(self) -> float:
        return self.embedding_count / self.embedding_seconds if self.embedding_seconds > 0 else float("nan")

    @property
    def ci_str(self) -> str:
        if self.ci_low is None or self.ci_high is None:
            return "-"
        return f"[{self.ci_low:.4f}, {self.ci_high:.4f}]"


def result_to_record(model_name: str, dataset: str, result: EvalResult, **extra) -> dict:
    """Flat dict for a JSON Lines / analysis record (no timestamp - that's
    added by append_json_record). Any keyword args (e.g. batch=...) are merged
    in."""
    record = {
        "model": model_name,
        "dataset": dataset,
        "accuracy": result.accuracy,
        "correct": result.correct,
        "total": result.total,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "mrr": result.mrr,
        "embedding_count": result.embedding_count,
        "embedding_seconds": result.embedding_seconds,
        "images_per_sec": result.images_per_sec,
    }
    record.update(extra)
    return record


def append_markdown_result(path: str, model_name: str, dataset: str, result: EvalResult) -> None:
    """Appends one row to a markdown table at `path`, creating the file (with
    header) if it doesn't exist yet. Never overwrites prior rows, so the same
    path can be reused across multiple evaluation runs to build up one report.

    Safe for concurrent runs (e.g. the two crop modes on different GPUs)
    appending to the same file: the header decision uses the append handle's
    own position, and header+row go out in a single append-mode write, so
    rows are never lost or torn - the worst case is a duplicated header if
    two processes create the file simultaneously, which readers (including
    src/collect_results.py) already skip as a non-data line."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ims = result.images_per_sec
    ims_str = f"{ims:.1f}" if ims == ims else "-"  # NaN != NaN
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = (
        f"| {model_name} | {dataset} | {result.accuracy:.4f} | {result.ci_str} | "
        f"{result.correct}/{result.total} | {ims_str} | {timestamp} |\n"
    )
    with open(out_path, "a", encoding="utf-8") as f:
        if f.tell() == 0:
            row = "# Evaluation Results\n\n" + MARKDOWN_HEADER + row
        f.write(row)
    print(f"Appended result to {out_path}")


def write_timing_json(path: str, record: dict) -> None:
    """Writes (overwriting) a single-object timing.json for a training run -
    wall-clock time, epochs, throughput, device - next to its checkpoints, so
    src/collect_results.py can build the training-time table. Unlike the
    append-only results logs, this is one file per run, rewritten each call."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_record = {**record, "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    out_path.write_text(json.dumps(full_record, indent=2))
    print(f"Wrote {out_path}")


def append_json_record(path: str, record: dict) -> None:
    """Appends one JSON object (with an added timestamp) as a line to a
    JSON Lines file at `path`, creating the file/parent dirs if needed. Safe
    to call repeatedly across runs/processes - each call only adds a line,
    never rewrites prior ones."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_record = {**record, "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(full_record) + "\n")
