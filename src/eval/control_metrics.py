"""Control-relevant error metrics, stratified by curvature bin.

Roadmap step four. IoU answers "how many lane pixels overlap", which is not the
question the controller asks. A model can win on overlap and still misplace the
centreline the MPC tracks, because IoU is dominated by the many easy pixels near the
bumper while the controller is most sensitive to the geometry further ahead.

This module measures the quantities the controller actually consumes, comparing the
geometry recovered from a predicted mask against the geometry recovered from the
ground-truth mask through the *same* pipeline (centreline, projection, spline). Errors
are reported per curvature bin, and **detection failures are counted rather than
dropped**: a frame where no ego lane can be formed from the prediction is a control
failure, and silently skipping it would flatter the model.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from src.eval.metrics import assign_bins
from src.geometry.road_geometry import RoadGeometry

FloatArray = np.ndarray


@dataclass(frozen=True)
class ControlErrors:
    """Per-frame error between predicted and ground-truth road geometry.

    Attributes:
        lateral_offset_err_m: Absolute lateral-offset error at the vehicle, in metres.
        heading_err_rad: Absolute heading error, in radians.
        curvature_err_1pm: Absolute signed-curvature error per preview distance, in
            ``1/m``; ``nan`` where either geometry does not reach that distance.
    """

    lateral_offset_err_m: float
    heading_err_rad: float
    curvature_err_1pm: FloatArray


def control_errors(pred: RoadGeometry, truth: RoadGeometry) -> ControlErrors:
    """Absolute errors between a predicted and a reference road geometry.

    Args:
        pred: Geometry recovered from the predicted mask.
        truth: Geometry recovered from the ground-truth mask.

    Returns:
        The per-frame :class:`ControlErrors`. Curvature entries are ``nan`` where
        either side lacks a value at that preview distance, so they can be excluded
        from the mean rather than counted as zero error.
    """
    return ControlErrors(
        lateral_offset_err_m=abs(pred.lateral_offset_m - truth.lateral_offset_m),
        heading_err_rad=abs(pred.heading_error_rad - truth.heading_error_rad),
        curvature_err_1pm=np.abs(
            pred.preview_curvature_1pm - truth.preview_curvature_1pm
        ),
    )


@dataclass
class BinControlSummary:
    """Aggregated control errors for one curvature bin.

    Attributes:
        offset_mae_m: Mean absolute lateral-offset error, metres.
        heading_mae_deg: Mean absolute heading error, degrees.
        curvature_mae_1pm: Mean absolute curvature error per preview distance.
        detected: Frames where both prediction and truth yielded a geometry.
        failed: Frames where the prediction yielded no ego lane but truth did.
        detection_rate: ``detected / (detected + failed)``, or ``nan`` if neither.
    """

    offset_mae_m: float
    heading_mae_deg: float
    curvature_mae_1pm: FloatArray
    detected: int
    failed: int
    detection_rate: float


@dataclass
class StratifiedControlMetric:
    """Accumulate control errors per curvature bin over a dataset.

    Args:
        bin_edges: ``K + 1`` curvature bin edges, matching the stratification.
        num_previews: Number of preview distances reported per frame.
    """

    bin_edges: list[float]
    num_previews: int = 3
    _offset: list[list[float]] = field(init=False)
    _heading: list[list[float]] = field(init=False)
    _kappa: list[list[FloatArray]] = field(init=False)
    _failed: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self._n_bins = len(self.bin_edges) - 1
        self.reset()

    def reset(self) -> None:
        self._offset = [[] for _ in range(self._n_bins)]
        self._heading = [[] for _ in range(self._n_bins)]
        self._kappa = [[] for _ in range(self._n_bins)]
        self._failed = np.zeros(self._n_bins, dtype=np.int64)

    def _bin_of(self, kappa: float) -> int:
        return int(assign_bins(np.asarray([kappa], dtype=np.float64), self.bin_edges)[0])

    def update(self, errors: ControlErrors, frame_kappa: float) -> None:
        """Record one successfully measured frame.

        Args:
            errors: Per-frame errors from :func:`control_errors`.
            frame_kappa: The frame's stratification curvature (bin assignment).
        """
        b = self._bin_of(frame_kappa)
        self._offset[b].append(errors.lateral_offset_err_m)
        self._heading[b].append(errors.heading_err_rad)
        self._kappa[b].append(np.asarray(errors.curvature_err_1pm, dtype=np.float64))

    def update_failure(self, frame_kappa: float) -> None:
        """Record a frame where no ego lane could be recovered from the prediction.

        Args:
            frame_kappa: The frame's stratification curvature (bin assignment).
        """
        self._failed[self._bin_of(frame_kappa)] += 1

    def compute(self) -> tuple[BinControlSummary, list[BinControlSummary]]:
        """Summarize the accumulated errors.

        Returns:
            ``(overall, per_bin)``. Bins with no measured frames report ``nan`` means
            so they are not mistaken for perfect. Curvature means ignore ``nan``
            entries, which mark preview distances neither side reached.
        """
        per_bin = [self._summarize(b) for b in range(self._n_bins)]
        overall = self._summarize_pooled()
        return overall, per_bin

    def _summarize(self, b: int) -> BinControlSummary:
        return self._build(
            self._offset[b], self._heading[b], self._kappa[b], int(self._failed[b])
        )

    def _summarize_pooled(self) -> BinControlSummary:
        offset = [v for bin_vals in self._offset for v in bin_vals]
        heading = [v for bin_vals in self._heading for v in bin_vals]
        kappa = [v for bin_vals in self._kappa for v in bin_vals]
        return self._build(offset, heading, kappa, int(self._failed.sum()))

    def _build(
        self,
        offset: list[float],
        heading: list[float],
        kappa: list[FloatArray],
        failed: int,
    ) -> BinControlSummary:
        detected = len(offset)
        total = detected + failed
        if detected == 0:
            kappa_mae = np.full(self.num_previews, np.nan)
        else:
            stacked = np.vstack(kappa)
            # A preview column that is all-nan (neither side reached that distance)
            # legitimately means "no measurement", so let nanmean return nan quietly.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                kappa_mae = np.nanmean(stacked, axis=0)
        return BinControlSummary(
            offset_mae_m=float(np.mean(offset)) if detected else float("nan"),
            heading_mae_deg=float(np.degrees(np.mean(heading)))
            if detected
            else float("nan"),
            curvature_mae_1pm=np.asarray(kappa_mae, dtype=np.float64),
            detected=detected,
            failed=failed,
            detection_rate=float(detected / total) if total else float("nan"),
        )
