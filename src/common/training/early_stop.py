"""
Two-stage early-stopping policy.

Stage one monitors the primary metric with `patience`. Without a secondary metric,
exhausting it stops the run (single-stage). With one, it instead arms stage two, which
monitors the secondary metric with `secondary_patience` and stops as soon as the primary
metric drifts more than `drift_tolerance` below its best.

The policy tracks the operating point (best secondary value among evaluations within the
drift band of the primary best) and which evaluated epochs currently lie in that band.
"""

from dataclasses import dataclass, field
from typing import Any

from .config import EarlyStopCfg


@dataclass
class EvalOutcome:
    """
    Decisions taken by the policy for one evaluation.

    Attributes
    ----------
    improved : bool
        The primary metric reached a new best; write `best_ckpt.pt`.
    band_save : bool
        This epoch lies within the drift band; write `band_ep{epoch}.pt`.
    band_drop : list[int]
        Epochs whose band checkpoints fell out of the band.
    annotation : str
        Stage-two suffix for the epoch's log line; empty when nothing to report.
    stop : str | None
        Stop line when training should end after this evaluation, else `None`.

    """

    improved: bool = False
    band_save: bool = False
    band_drop: list[int] = field(default_factory=list)
    annotation: str = ""
    stop: str | None = None


class TwoStageEarlyStopping:
    """Track early-stopping state across evaluations and decide when to stop."""

    def __init__(self, cfg: EarlyStopCfg) -> None:
        """
        Initialise the policy state.

        Parameters
        ----------
        cfg : EarlyStopCfg
            Early-stopping configuration.

        """
        self.cfg = cfg
        self.split = cfg.split
        self.metric = cfg.metric
        self.mode = cfg.mode
        self.secondary_metric = cfg.secondary_metric
        self.secondary_mode = cfg.secondary_mode
        self.patience = cfg.patience
        self.secondary_patience = (
            cfg.secondary_patience
            if cfg.secondary_patience is not None
            else cfg.patience
        )

        self.worst = float("inf") if self.mode == "min" else -float("inf")
        self.secondary_worst = (
            float("inf") if self.secondary_mode == "min" else -float("inf")
        )

        self.best_metric = self.worst
        self.best_metrics: dict[str, Any] = {}
        self.patience_left = self.patience

        self.stage2_armed = False
        self.secondary_best = self.secondary_worst
        self.secondary_left = self.secondary_patience
        self.inadmissible = False

        self.op_value = self.secondary_worst
        self.op_primary = self.worst
        self.op_metrics: dict[str, Any] = {}

        # Mapping an evaluated epoch within the drift band to its primary value
        self.band_primary: dict[int, float] = {}

    @staticmethod
    def _better(a: float, b: float, mode: str) -> bool:
        """Return whether `a` is strictly better than `b` under `mode`."""
        return a < b if mode == "min" else a > b

    def _band_line(self) -> float:
        """Return the worst primary value still within `drift_tolerance` of the best."""
        tol = self.cfg.drift_tolerance
        if self.mode == "min":
            return self.best_metric * (1.0 + tol)
        return self.best_metric * (1.0 - tol)

    def _within_band(self, primary: float) -> bool:
        """Return whether `primary` lies within the drift band of the best."""
        line = self._band_line()
        return primary <= line if self.mode == "min" else primary >= line

    def observe(
        self,
        epoch: int,
        val_metrics: dict[str, float],
        test_metrics: dict[str, float],
    ) -> EvalOutcome:
        """
        Update the policy with one evaluation and return the resulting decisions.

        Parameters
        ----------
        epoch : int
            Epoch of the evaluation.
        val_metrics : dict[str, float]
            Validation metrics at this epoch.
        test_metrics : dict[str, float]
            Test metrics at this epoch.

        Returns
        -------
        EvalOutcome
            Checkpoint actions, log annotation, and stop decision for this evaluation.

        """
        out = EvalOutcome()
        selected = val_metrics if self.split == "val" else test_metrics
        split_record = {
            **{f"test_{k}": v for k, v in test_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }

        primary = selected[self.metric]
        out.improved = self._better(primary, self.best_metric, self.mode)
        if out.improved:
            self.best_metric = primary
            self.best_metrics = {
                "epoch": epoch,
                "early_stop_split": self.split,
                "selected_metric": self.metric,
                "selected_value": primary,
                **split_record,
            }
            self.patience_left = self.patience
        else:
            self.patience_left -= 1

        if self.cfg.save_eligible_checkpoints:
            if out.improved:
                # A new best moves the drift line, so delete band checkpoints that were
                # eligible before but now fall outside it
                for ep, p in list(self.band_primary.items()):
                    if not self._within_band(p):
                        out.band_drop.append(ep)
                        del self.band_primary[ep]
            if self._within_band(primary):
                out.band_save = True
                self.band_primary[epoch] = primary

        if self.secondary_metric is not None:
            self._update_operating_point(epoch, primary, selected, split_record)

        stage2_stop: str | None = None
        if self.stage2_armed and self.secondary_metric is not None:
            drifted = self._better(self._band_line(), primary, self.mode)
            secondary = selected[self.secondary_metric]
            if self._better(secondary, self.secondary_best, self.secondary_mode):
                self.secondary_best = secondary
                self.secondary_left = self.secondary_patience
            else:
                self.secondary_left -= 1
            out.annotation = (
                f"  [stage2 {self.secondary_metric}={secondary:.5f}"
                f" best={self.secondary_best:.5f} left={max(self.secondary_left, 0)}"
                f"{' DRIFTED' if drifted else ''}]"
            )
            if drifted or self.secondary_left <= 0:
                stage2_stop = "primary drift" if drifted else "secondary plateau"
        elif self.secondary_metric is not None and self.patience_left <= 0:
            # The primary metric has plateaued.
            # Either abandon (check admissibility_floor) or arm stage two.
            floor = self.cfg.admissibility_floor
            if floor is not None and self._better(floor, self.best_metric, self.mode):
                self.inadmissible = True
                out.annotation = (
                    f"  [ABANDONED: {self.metric}={self.best_metric:.5f} "
                    f"plateaued outside floor {floor:.5f}]"
                )
            else:
                self.stage2_armed = True
                # secondary_best is first set and updated from here on out
                self.secondary_best = selected[self.secondary_metric]
                self.secondary_left = self.secondary_patience
                out.annotation = (
                    f"  [stage2 ARMED at {self.secondary_metric}"
                    f"={self.secondary_best:.5f}]"
                )

        out.stop = self._stop_line(epoch, stage2_stop)
        return out

    def _update_operating_point(
        self,
        epoch: int,
        primary: float,
        selected: dict[str, float],
        split_record: dict[str, float],
    ) -> None:
        """
        Track the best secondary value among evaluations within the drift band.

        A new primary best raises the band line, which can disqualify the operating
        point recorded so far. Drop it when that happens; the current epoch is eligible
        by construction and the test below re-establishes it.
        """
        assert self.secondary_metric is not None
        # op_* store the current operating point, if any
        # discards it if it has fallen out the band, so the current epoch can replace it
        if self.op_metrics and not self._within_band(self.op_primary):
            self.op_value = self.secondary_worst
        sec = selected[self.secondary_metric]
        if self._within_band(primary) and self._better(
            sec, self.op_value, self.secondary_mode
        ):
            self.op_value = sec
            self.op_primary = primary
            self.op_metrics = {
                "epoch": epoch,
                "early_stop_split": self.split,
                "selected_metric": self.secondary_metric,
                "selected_value": sec,
                "primary_metric": self.metric,
                "primary_value": primary,
                **split_record,
            }

    def _stop_line(self, epoch: int, stage2_stop: str | None) -> str | None:
        """
        Return the stop line if training should end at `epoch`, else `None`.

        Single-stage: `patience` on the primary metric terminates. Two-stage: it only
        arms, and stage two owns termination from then on.
        """
        best = f"best {self.split} {self.metric}={self.best_metric:.4f}"
        if self.secondary_metric is None:
            if self.patience_left <= 0:
                return f"early stop at epoch {epoch}; {best}"
            return None
        if self.inadmissible:
            return (
                f"abandoned at epoch {epoch} (inadmissible); {best} "
                f"outside floor {self.cfg.admissibility_floor:.4f}"
            )
        if stage2_stop is not None:
            return (
                f"early stop at epoch {epoch} ({stage2_stop}); {best}, "
                f"stage-2 best {self.secondary_metric}={self.secondary_best:.4f}"
            )
        return None

    def result(self) -> dict[str, Any]:
        """
        Return the run summary written to `metrics.json`.

        Returns
        -------
        dict[str, Any]
            Metrics at the primary best epoch, plus `abandoned`/`admissibility_floor`
            when the run was abandoned, `operating_point` when one was recorded, and
            `band_checkpoints` when band checkpoints are on.

        """
        summary = dict(self.best_metrics)
        if self.inadmissible:
            summary["abandoned"] = True
        if self.op_metrics:
            summary["operating_point"] = self.op_metrics
        if self.cfg.save_eligible_checkpoints:
            summary["band_checkpoints"] = sorted(self.band_primary)
        return summary
