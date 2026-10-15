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
    ci_low: float | None = None
    ci_high: float | None = None
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
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ims = result.images_per_sec
    ims_str = f"{ims:.1f}" if ims == ims else "-"
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
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_record = {**record, "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    out_path.write_text(json.dumps(full_record, indent=2))
    print(f"Wrote {out_path}")


def append_json_record(path: str, record: dict) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_record = {**record, "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(full_record) + "\n")
