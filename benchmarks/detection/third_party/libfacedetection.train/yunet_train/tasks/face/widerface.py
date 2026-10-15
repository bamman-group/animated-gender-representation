from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import FaceAnnotation, FaceRecord


def _parse_annotation_line(
    line: str,
    *,
    min_size: int | None,
    test_mode: bool,
    keypoint_count: int = 5,
) -> FaceAnnotation:
    values = [float(value) for value in line.strip().split()]
    if len(values) < 4:
        raise ValueError(f"Expected at least 4 bbox values, got {len(values)}: {line!r}")

    bbox = np.array(values[0:4], dtype=np.float32)
    keypoints = np.zeros((keypoint_count, 3), dtype=np.float32)
    ignore = False

    if min_size is not None:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width < min_size or height < min_size:
            ignore = True

    if len(values) > 4:
        if len(values) > 5:
            keypoints = np.array(
                values[4 : 4 + keypoint_count * 3],
                dtype=np.float32,
            ).reshape((keypoint_count, 3))
            for point in keypoints:
                if (point == -1).all():
                    point[2] = 0.0
                else:
                    if point[2] < 0:
                        raise ValueError(f"Invalid keypoint weight in line: {line!r}")
                    point[2] = 1.0
        elif not ignore:
            ignore = values[4] == 1
    elif not test_mode:
        raise ValueError(f"Missing train annotation flags/keypoints: {line!r}")

    return FaceAnnotation(bbox=bbox, keypoints=keypoints, ignore=ignore)


def parse_labelv2_file(
    ann_file: str | Path,
    *,
    min_size: int | None = None,
    test_mode: bool = False,
) -> list[FaceRecord]:
    ann_path = Path(ann_file)
    records: list[FaceRecord] = []

    current_name: str | None = None
    current_width: int | None = None
    current_height: int | None = None
    current_annotations: list[FaceAnnotation] = []

    def flush_current() -> None:
        if current_name is None or current_width is None or current_height is None:
            return
        if current_annotations or test_mode:
            records.append(
                FaceRecord(
                    filename=current_name,
                    width=current_width,
                    height=current_height,
                    annotations=tuple(current_annotations),
                )
            )

    with ann_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("#"):
                flush_current()
                parts = line[1:].strip().split()
                if len(parts) != 3:
                    raise ValueError(f"Invalid labelv2 image header: {line!r}")
                current_name = parts[0]
                current_width = int(parts[1])
                current_height = int(parts[2])
                current_annotations = []
                continue

            if current_name is None:
                raise ValueError(f"Annotation appears before image header: {line!r}")

            current_annotations.append(
                _parse_annotation_line(
                    line,
                    min_size=min_size,
                    test_mode=test_mode,
                )
            )

    flush_current()
    return records

