"""Tests for the CurveLanes annotation parser."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.data.curvelanes import (
    Lane,
    build_frame,
    index_split,
    label_stem,
    parse_lines_json,
)


def _write_label(path: Path, lines: list[list[tuple[float, float]]]) -> None:
    payload = {"Lines": [[{"x": str(x), "y": str(y)} for x, y in line] for line in lines]}
    path.write_text(json.dumps(payload))


def test_parse_basic_two_lanes(tmp_path: Path) -> None:
    label = tmp_path / "frame.lines.json"
    _write_label(label, [[(0.0, 554.86), (394.14, 473.91)], [(317.58, 659.0), (719.58, 415.04)]])

    lanes = parse_lines_json(label)

    assert len(lanes) == 2
    assert lanes[0].points.dtype == np.float64
    np.testing.assert_allclose(lanes[0].points, [[0.0, 554.86], [394.14, 473.91]])
    assert lanes[0].num_points == 2


def test_string_coords_are_cast_to_float(tmp_path: Path) -> None:
    label = tmp_path / "frame.lines.json"
    _write_label(label, [[(1.5, 2.5), (3.0, 4.0)]])
    (x0, y0), _ = parse_lines_json(label)[0].points
    assert isinstance(float(x0), float) and (x0, y0) == (1.5, 2.5)


def test_edge_points_are_kept_not_clipped(tmp_path: Path) -> None:
    # Points on/beyond the frame edge must survive parsing untouched.
    label = tmp_path / "frame.lines.json"
    _write_label(label, [[(0.0, 100.0), (-5.0, 200.0), (2560.0, 300.0)]])
    pts = parse_lines_json(label)[0].points
    assert pts[1, 0] == -5.0 and pts[2, 0] == 2560.0


def test_degenerate_single_point_lane_dropped(tmp_path: Path) -> None:
    label = tmp_path / "frame.lines.json"
    _write_label(label, [[(10.0, 20.0)], [(0.0, 1.0), (2.0, 3.0)]])
    lanes = parse_lines_json(label)
    assert len(lanes) == 1  # single-point polyline removed


def test_empty_lines_yields_empty_list(tmp_path: Path) -> None:
    label = tmp_path / "frame.lines.json"
    label.write_text(json.dumps({"Lines": []}))
    assert parse_lines_json(label) == []


def test_missing_lines_key_yields_empty_list(tmp_path: Path) -> None:
    label = tmp_path / "frame.lines.json"
    label.write_text(json.dumps({}))
    assert parse_lines_json(label) == []


def test_malformed_point_raises(tmp_path: Path) -> None:
    label = tmp_path / "frame.lines.json"
    label.write_text(json.dumps({"Lines": [[{"x": "1.0", "y": "notanumber"}, {"x": "2", "y": "3"}]]}))
    with pytest.raises(ValueError):
        parse_lines_json(label)


def test_lane_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        Lane(points=np.zeros((3, 3)))
    with pytest.raises(ValueError):
        Lane(points=np.zeros((1, 2)))  # too few points


def test_label_stem_strips_compound_suffix() -> None:
    assert label_stem(Path("dir/abc123.lines.json")) == "abc123"
    assert label_stem(Path("plain.json")) == "plain"


def test_index_split_pairs_and_reports_mismatches(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()

    (images / "a.jpg").touch()
    (images / "b.jpg").touch()
    (images / "orphan_img.jpg").touch()
    _write_label(labels / "a.lines.json", [[(0.0, 0.0), (1.0, 1.0)]])
    _write_label(labels / "b.lines.json", [[(0.0, 0.0), (1.0, 1.0)]])
    _write_label(labels / "orphan_lbl.lines.json", [[(0.0, 0.0), (1.0, 1.0)]])

    idx = index_split(images, labels)

    assert [p[0].stem for p in idx.pairs] == ["a", "b"]
    assert [p.stem for p in idx.images_without_label] == ["orphan_img"]
    assert [label_stem(p) for p in idx.labels_without_image] == ["orphan_lbl"]


def test_build_frame_reads_native_size(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (1570, 660)).save(image_path)
    label_path = tmp_path / "frame.lines.json"
    _write_label(label_path, [[(0.0, 100.0), (1000.0, 200.0)]])

    frame = build_frame(image_path, label_path)

    assert (frame.width, frame.height) == (1570, 660)
    assert frame.num_lanes == 1
