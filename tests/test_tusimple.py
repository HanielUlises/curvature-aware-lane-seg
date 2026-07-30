"""Tests for the TuSimple label parser."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.data.tusimple import iter_label_frames, parse_label_line


def _line(lanes, rows, raw_file="clips/0313-1/6040/20.jpg"):
    return json.dumps({"lanes": lanes, "h_samples": rows, "raw_file": raw_file})


def test_absent_rows_are_dropped():
    rows = [240, 250, 260, 270, 280]
    # -2 marks rows where the lane is not present.
    frame = parse_label_line(_line([[-2, -2, 632, 625, 617]], rows), min_points=3)
    assert len(frame.lanes) == 1
    np.testing.assert_allclose(
        frame.lanes[0], [[632, 260], [625, 270], [617, 280]]
    )


def test_lanes_with_too_few_points_are_discarded():
    rows = [240, 250, 260, 270]
    frame = parse_label_line(_line([[-2, -2, -2, 600]], rows), min_points=4)
    assert frame.lanes == []


def test_points_ordered_by_increasing_row():
    rows = [240, 250, 260, 270, 280]
    frame = parse_label_line(_line([[600, 601, 602, 603, 604]], rows), min_points=4)
    ys = frame.lanes[0][:, 1]
    assert np.all(np.diff(ys) > 0)


def test_raw_file_preserved():
    frame = parse_label_line(_line([[1, 2, 3, 4]], [240, 250, 260, 270]))
    assert frame.raw_file == "clips/0313-1/6040/20.jpg"


def test_iter_label_frames_filters_by_lane_count(tmp_path):
    rows = [240, 250, 260, 270]
    good = _line([[600, 601, 602, 603], [700, 701, 702, 703]], rows)
    only_one = _line([[600, 601, 602, 603]], rows)
    (tmp_path / "label_data_0313.json").write_text(good + "\n" + only_one + "\n\n")
    frames = list(iter_label_frames(tmp_path, min_lanes=2))
    assert len(frames) == 1
    assert len(frames[0].lanes) == 2


def test_iter_label_frames_requires_label_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(iter_label_frames(tmp_path))
