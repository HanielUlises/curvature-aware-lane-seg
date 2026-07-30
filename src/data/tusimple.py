"""Parser for TuSimple lane annotations.

TuSimple is used here for camera calibration rather than training. Its footage comes
from one vehicle fleet with a consistent camera, which is what makes a single
calibration identifiable; CurveLanes aggregates many sources and does not admit one
(see :mod:`src.geometry.calibration`). TuSimple also annotates lanes up to the horizon
region, so the geometry needed for calibration is actually observed.

Each line of a ``label_data_*.json`` file is one frame:

```
{"lanes": [[x, ...], ...], "h_samples": [y, ...], "raw_file": "clips/.../20.jpg"}
```

Lane entries give the column at each row in ``h_samples``, with a non-positive value
marking a row where that lane is absent. Coordinates are native pixels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

FloatArray = np.ndarray

# TuSimple frames are all this size.
TUSIMPLE_IMAGE_SIZE = (1280, 720)
# A lane needs at least this many annotated rows to be worth fitting.
DEFAULT_MIN_POINTS = 4


@dataclass(frozen=True)
class TuSimpleFrame:
    """One annotated TuSimple frame.

    Attributes:
        raw_file: Clip-relative path of the annotated image.
        lanes: Lane polylines, each ``(N, 2)`` as ``(x, y)`` in native pixels,
            ordered by increasing row.
    """

    raw_file: str
    lanes: list[FloatArray]


def parse_label_line(line: str, min_points: int = DEFAULT_MIN_POINTS) -> TuSimpleFrame:
    """Parse one JSON line into a frame of lane polylines.

    Args:
        line: One line of a ``label_data_*.json`` file.
        min_points: Minimum annotated rows for a lane to be kept.

    Returns:
        The parsed :class:`TuSimpleFrame`; lanes with too few points are dropped.
    """
    record = json.loads(line)
    rows = np.asarray(record["h_samples"], dtype=np.float64)
    lanes: list[FloatArray] = []
    for columns in record["lanes"]:
        xs = np.asarray(columns, dtype=np.float64)
        present = xs > 0.0  # non-positive marks an absent row
        if int(present.sum()) < min_points:
            continue
        lanes.append(np.column_stack([xs[present], rows[present]]))
    return TuSimpleFrame(raw_file=str(record["raw_file"]), lanes=lanes)


def iter_label_frames(
    label_dir: Path,
    min_lanes: int = 2,
    min_points: int = DEFAULT_MIN_POINTS,
) -> Iterator[TuSimpleFrame]:
    """Yield annotated frames from every ``label_data_*.json`` under a directory.

    Args:
        label_dir: Directory holding the label files (the TuSimple ``train_set``).
        min_lanes: Skip frames with fewer surviving lanes than this.
        min_points: Minimum annotated rows for a lane to be kept.

    Yields:
        Frames in file order.

    Raises:
        FileNotFoundError: If no label files are present.
    """
    files = sorted(Path(label_dir).glob("label_data_*.json"))
    if not files:
        raise FileNotFoundError(f"no label_data_*.json under {label_dir}")
    for path in files:
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                frame = parse_label_line(line, min_points)
                if len(frame.lanes) >= min_lanes:
                    yield frame
