"""Parser for CurveLanes polyline annotations.

Loads raw CurveLanes ``*.lines.json`` annotation files into a typed intermediate
representation. This module performs **no rasterization and no resizing** — it
only reads polylines (and, on request, the native image resolution) so that the
rasterization and curvature stages downstream operate on faithful native-frame
geometry.

CurveLanes label schema (one file per frame, ``<stem>.lines.json``)::

    {"Lines": [
        [{"x": "0.0", "y": "554.86"}, {"x": "394.14", "y": "473.91"}, ...],
        [{"x": "317.58", "y": "659.0"}, ...],
        ...
    ]}

Coordinates are strings in native pixel space and are cast to ``float``. Points
may lie on or beyond the frame edge (e.g. ``x == 0.0``); such points are kept
verbatim — clipping is the rasterizer's responsibility, not the parser's. The
native resolution is *not* stored in the label file; it is read from the image
header only when a full :class:`FrameAnnotation` is built.

The dataset ships a mix of native resolutions (predominantly 2560x1440, with a
minority at 1570x660). Each :class:`FrameAnnotation` therefore records its own
``width``/``height`` so that no downstream stage assumes a single resolution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

LABEL_SUFFIX = ".lines.json"
IMAGE_SUFFIX = ".jpg"

# Minimum points for a polyline to define a segment; fewer is degenerate.
_MIN_POINTS = 2


@dataclass(frozen=True)
class Lane:
    """A single lane polyline in native pixel coordinates.

    Attributes:
        points: Array of shape ``(N, 2)``, columns ``(x, y)``, ``N >= 2``.
            Points are in native image pixels and are ordered as annotated;
            no reordering or clipping is applied.
    """

    points: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError(f"Lane.points must have shape (N, 2), got {self.points.shape}")
        if self.points.shape[0] < _MIN_POINTS:
            raise ValueError(f"Lane needs >= {_MIN_POINTS} points, got {self.points.shape[0]}")

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])


@dataclass(frozen=True)
class FrameAnnotation:
    """All lane annotations for a single frame, with native resolution.

    Attributes:
        image_path: Path to the source ``.jpg``.
        label_path: Path to the source ``.lines.json``.
        width: Native image width in pixels.
        height: Native image height in pixels.
        lanes: Parsed lane polylines (degenerate lanes already removed).
    """

    image_path: Path
    label_path: Path
    width: int
    height: int
    lanes: list[Lane]

    @property
    def num_lanes(self) -> int:
        return len(self.lanes)


def parse_lines_json(label_path: Path) -> list[Lane]:
    """Parse a CurveLanes ``*.lines.json`` file into a list of lanes.

    String coordinates are cast to ``float``. Polylines with fewer than two
    points are silently dropped as degenerate; an empty ``Lines`` array yields
    an empty list.

    Args:
        label_path: Path to a ``*.lines.json`` annotation file.

    Returns:
        A list of :class:`Lane`, one per valid annotated polyline.

    Raises:
        ValueError: If the JSON is malformed or a coordinate is not numeric.
    """
    with Path(label_path).open("r") as handle:
        payload = json.load(handle)

    raw_lines = payload.get("Lines", [])
    if not isinstance(raw_lines, list):
        raise ValueError(f"{label_path}: 'Lines' must be a list, got {type(raw_lines).__name__}")

    lanes: list[Lane] = []
    for line in raw_lines:
        try:
            points = np.array(
                [(float(pt["x"]), float(pt["y"])) for pt in line],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{label_path}: malformed point in a polyline") from exc

        # Reshape guards the empty-polyline case so shape stays (0, 2).
        points = points.reshape(-1, 2)
        if points.shape[0] < _MIN_POINTS:
            continue
        lanes.append(Lane(points=points))

    return lanes


def read_image_size(image_path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` from an image header without full decode.

    Args:
        image_path: Path to the image file.

    Returns:
        The native ``(width, height)`` in pixels.
    """
    from PIL import Image  # local import keeps parsing usable without PIL

    with Image.open(image_path) as image:
        width, height = image.size
    return int(width), int(height)


def build_frame(image_path: Path, label_path: Path) -> FrameAnnotation:
    """Build a :class:`FrameAnnotation` from a paired image and label.

    Args:
        image_path: Path to the frame image (``.jpg``).
        label_path: Path to the frame annotation (``.lines.json``).

    Returns:
        The fully populated :class:`FrameAnnotation`, including native size.
    """
    lanes = parse_lines_json(label_path)
    width, height = read_image_size(image_path)
    return FrameAnnotation(
        image_path=Path(image_path),
        label_path=Path(label_path),
        width=width,
        height=height,
        lanes=lanes,
    )


def label_stem(label_path: Path) -> str:
    """Return the frame stem for a ``*.lines.json`` label path.

    ``abc.lines.json`` -> ``abc``. Falls back to the plain stem for any label
    that does not carry the compound ``.lines.json`` suffix.
    """
    name = Path(label_path).name
    if name.endswith(LABEL_SUFFIX):
        return name[: -len(LABEL_SUFFIX)]
    return Path(label_path).stem


@dataclass(frozen=True)
class DatasetIndex:
    """Result of pairing an images directory with a labels directory.

    Attributes:
        pairs: ``(image_path, label_path)`` tuples sharing a stem, sorted by stem.
        images_without_label: Image paths whose stem has no matching label.
        labels_without_image: Label paths whose stem has no matching image.
    """

    pairs: list[tuple[Path, Path]]
    images_without_label: list[Path]
    labels_without_image: list[Path]


def index_split(images_dir: Path, labels_dir: Path) -> DatasetIndex:
    """Pair images with labels by shared stem within one dataset split.

    Args:
        images_dir: Directory of ``<stem>.jpg`` images.
        labels_dir: Directory of ``<stem>.lines.json`` labels. May be nested;
            labels are discovered recursively.

    Returns:
        A :class:`DatasetIndex` with matched pairs and both unmatched sets, so
        that callers can report and act on any image/label mismatch rather than
        silently dropping frames.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    images_by_stem = {p.stem: p for p in sorted(images_dir.rglob(f"*{IMAGE_SUFFIX}"))}
    labels_by_stem = {
        label_stem(p): p for p in sorted(labels_dir.rglob(f"*{LABEL_SUFFIX}"))
    }

    common = sorted(images_by_stem.keys() & labels_by_stem.keys())
    pairs = [(images_by_stem[stem], labels_by_stem[stem]) for stem in common]

    images_without_label = [
        images_by_stem[stem] for stem in sorted(images_by_stem.keys() - labels_by_stem.keys())
    ]
    labels_without_image = [
        labels_by_stem[stem] for stem in sorted(labels_by_stem.keys() - images_by_stem.keys())
    ]
    return DatasetIndex(
        pairs=pairs,
        images_without_label=images_without_label,
        labels_without_image=labels_without_image,
    )
