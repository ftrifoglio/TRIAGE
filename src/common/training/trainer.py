"""
Generic trainer.

Per-epoch:
  1. ask the `NeighbourhoodSampler` for the propagation adjacency
  2. iterate the BPR sampler (even if loss is SSM)
  3. forward the encoder once per batch
  4. sum BPR/SSM + L2 [+ aux-losses]
  5. eval every N epochs, early-stop.
"""

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from .config import Config
from .data import InteractionDataset
from .early_stop import TwoStageEarlyStopping
from .encoder import LightGCNEncoder
from .evaluator import Evaluator
from .losses import (
    aurl_align_loss,
    aurl_uniform_loss,
    bpr_loss,
    l2_reg,
    l2_reg_squared,
    ssm_margin_loss,
)
from .registry import ModelBundle
from .sampler import BPRSampler


@dataclass
class BatchContext:
    """
    Per-batch context passed to the auxiliary losses.

    Attributes
    ----------
    users : torch.Tensor of shape (B,), int64
        Batch user idx.
    pos_items : torch.Tensor of shape (B,), int64
        Positive item idx for the batch.
    user_emb : torch.Tensor of shape (n_users, d), float32
        Full user embedding table for this forward pass.
    item_emb : torch.Tensor of shape (n_items, d), float32
        Full item embedding table for this forward pass.
    adj : torch.Tensor of shape (n_users + n_items, n_users + n_items), float32
        Sparse propagation adjacency used for the forward pass.

    """

    users: torch.Tensor
    pos_items: torch.Tensor
    user_emb: torch.Tensor
    item_emb: torch.Tensor
    adj: torch.Tensor


AuxLoss = Callable[[BatchContext], torch.Tensor]


class Trainer:
    """Train a `ModelBundle` and evaluate using an `Evaluator`."""

    def __init__(
        self,
        bundle: ModelBundle,
        dataset: InteractionDataset,
        evaluator: Evaluator,
        cfg: Config,
        device: str,
        out_dir: Path,
    ) -> None:
        """
        Set up the trainer.

        Parameters
        ----------
        bundle : ModelBundle
            Encoder, neighbourhood sampler, extra modules, and aux losses.
        dataset : InteractionDataset
            Dataset providing train edges for the BPR sampler.
        evaluator : Evaluator
            Evaluator used for periodic metric computation.
        cfg : Config
            Full run configuration.
        device : str
            Torch device to train on.
        out_dir : Path
            Output directory for logs, metrics, and checkpoints; created if missing.

        """
        self.bundle = bundle
        self.dataset = dataset
        self.evaluator = evaluator
        self.cfg = cfg
        self.device = device
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.bpr_sampler = BPRSampler(
            dataset, cfg.training.batch_size, cfg.seed, cfg.training.bpr_sampling
        )
        self._init_ssm()
        self._init_aurl()

        params = list(bundle.encoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=cfg.training.lr)

    def _init_ssm(self) -> None:
        """Set up the SSM negative-sampling distribution."""
        self.ssm_neg_probs: torch.Tensor | None = None
        self.ssm_n_neg = 0
        ssm = self.cfg.training.ssm
        if ssm is None:
            return
        self.ssm_n_neg = ssm.n_neg or self.cfg.training.batch_size
        # Item popularity over the full `train.txt`, so the head/tail partition is
        # invariant to `val_ratio`. When `val_ratio > 0`, this means val edges
        # to the sampling weights (a mild leak into the training loss).
        deg = torch.from_numpy(self.dataset.item_degrees).float()
        self.ssm_neg_probs = deg.pow(ssm.beta).to(self.device)

    def _init_aurl(self) -> None:
        """Set up the AURL popular/unpopular partitions."""
        self.aurl_user_pop: torch.Tensor | None = None
        self.aurl_item_pop: torch.Tensor | None = None
        if self.cfg.training.aurl is None:
            return
        # One cut for both users and items (AURL Section 5.1.1) matching the evaluator's
        pop_quantile = self.cfg.eval.head_quantile
        user_deg = np.zeros(self.dataset.n_users, dtype=np.int64)
        # User activity over the full `train.txt`, so the head/tail partition is
        # invariant to `val_ratio`. When `val_ratio > 0`, this means val edges
        # contribute to the degree counts (a mild leak into the training loss).
        for u, items in self.dataset.train_val_user_items.items():
            user_deg[u] = len(items)
        user_deg_t = torch.from_numpy(user_deg).to(self.device)
        user_cutoff = torch.quantile(user_deg_t.float(), pop_quantile)
        self.aurl_user_pop = user_deg_t >= user_cutoff
        # Ditto. Item popularity over the full `train.txt`.
        item_deg_t = torch.from_numpy(self.dataset.item_degrees).to(self.device)
        item_cutoff = torch.quantile(item_deg_t.float(), pop_quantile)
        self.aurl_item_pop = item_deg_t >= item_cutoff

    def _negatives(self, neg_cpu: torch.Tensor) -> torch.Tensor:
        """
        Return the batch negatives: SSM draws its own, otherwise the BPR sampler's.

        Parameters
        ----------
        neg_cpu : torch.Tensor of shape (B,), int64
            Negatives drawn by the BPR sampler, on CPU.

        Returns
        -------
        torch.Tensor of shape (N,), int64
            Negative item idx on the training device; `N = B` under BPR and
            `N = ssm.n_neg or batch_size` under SSM.

        """
        if self.cfg.training.ssm is None:
            return neg_cpu.to(self.device)
        # SSM passes the layer-0 embeddings of the batch negatives to the
        # regularisation loss:
        # https://github.com/wujcan/SSM-TensorFlow/blob/661767c/model/general_recommender/SSM.py#L248-L272
        # We draw them here, not in `ssm_margin_loss`, and ignore the BPR sampled
        # `neg_cpu`, so `i_neg` and `e0[n_u + neg]` are indexed using the same items
        # and go to the SSM loss and the "he2020" regulariser, respectively.
        assert self.ssm_neg_probs is not None
        return torch.multinomial(self.ssm_neg_probs, self.ssm_n_neg, replacement=True)

    def _rec_loss(
        self, u_e: torch.Tensor, i_pos: torch.Tensor, i_neg: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the recommendation loss: SSM when `training.ssm` is set, else BPR.

        Parameters
        ----------
        u_e : torch.Tensor of shape (B, d), float32
            Batch user embeddings.
        i_pos : torch.Tensor of shape (B, d), float32
            Batch positive item embeddings.
        i_neg : torch.Tensor of shape (N, d), float32
            Batch negative item embeddings; `N = B` under BPR.

        Returns
        -------
        torch.Tensor of shape (), float32
            Recommendation loss.

        """
        ssm = self.cfg.training.ssm
        if ssm is None:
            return bpr_loss(u_e, i_pos, i_neg)
        return ssm_margin_loss(u_e, i_pos, i_neg, ssm.tau, ssm.margin)

    def _aurl_loss(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        users: torch.Tensor,
        pos_items: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the weighted AURL alignment + uniformity terms.

        Parameters
        ----------
        user_emb : torch.Tensor of shape (n_users, d), float32
            Full user embedding table.
        item_emb : torch.Tensor of shape (n_items, d), float32
            Full item embedding table.
        users : torch.Tensor of shape (B,), int64
            Batch user idx.
        pos_items : torch.Tensor of shape (B,), int64
            Batch positive item idx.

        Returns
        -------
        torch.Tensor of shape (), float32
            `align_weight * align + uniform_weight * uniform`.

        """
        a = self.cfg.training.aurl
        assert a is not None
        assert self.aurl_user_pop is not None
        assert self.aurl_item_pop is not None
        align = aurl_align_loss(
            user_emb,
            item_emb,
            users,
            pos_items,
            self.aurl_user_pop,
            self.aurl_item_pop,
            a.kernel_mul,
            a.kernel_num,
            a.mmd_batch,
        )
        uniform = aurl_uniform_loss(
            user_emb, item_emb, users, pos_items, a.t, a.uniform_batch
        )
        return a.align_weight * align + a.uniform_weight * uniform

    def train(self) -> dict[str, Any]:
        """
        Run the training loop with periodic eval and early stopping.

        Returns
        -------
        dict[str, Any]
            Metrics of the split named by `training.early_stop.split` at the best epoch,
            augmented with the `epoch`, `early_stop_split`, and
            `selected_metric`/`selected_value`.

        """
        bundle = self.bundle
        cfg = self.cfg
        bundle.encoder.to(self.device)

        stopper = TwoStageEarlyStopping(cfg.training.early_stop)

        log_path = self.out_dir / "train.log"
        log_path.write_text("")
        trace_path = self.out_dir / "trace.jsonl"
        trace_path.write_text("")

        # Drop band checkpoints from a previous run in this folder
        for old in self.out_dir.glob("band_ep*.pt"):
            old.unlink()

        resolved = asdict(cfg)
        resolved["device_name"] = (
            torch.cuda.get_device_name(0)
            if self.device.startswith("cuda") and torch.cuda.is_available()
            else self.device
        )
        (self.out_dir / "config.json").write_text(
            json.dumps(resolved, indent=2, default=str) + "\n"
        )

        # Start epoch loop
        for epoch in range(cfg.training.epochs):
            bundle.encoder.train()

            adj = bundle.neighbourhood_sampler.sample(
                epoch, {"encoder": bundle.encoder}
            )
            adj = adj.to(self.device)

            t0 = time.time()
            sums = {"loss": 0.0, "rec": 0.0, "l2": 0.0, "aux": 0.0}
            n_batches = 0
            for users_cpu, pos_cpu, neg_cpu in self.bpr_sampler:
                users = users_cpu.to(self.device)
                pos = pos_cpu.to(self.device)

                u_emb_all, i_emb_all = bundle.encoder(adj)
                u_e = u_emb_all[users]
                i_pos = i_emb_all[pos]

                neg = self._negatives(neg_cpu)
                i_neg = i_emb_all[neg]
                rec = self._rec_loss(u_e, i_pos, i_neg)

                if cfg.training.reg_form == "he2020":
                    encoder = cast(LightGCNEncoder, bundle.encoder)
                    e0 = encoder.initial_table()
                    n_u = self.dataset.n_users
                    reg = cfg.training.l2 * l2_reg_squared(
                        e0[users], e0[n_u + pos], e0[n_u + neg]
                    )
                else:
                    reg = cfg.training.l2 * l2_reg(u_e, i_pos)

                ctx = BatchContext(
                    users=users,
                    pos_items=pos,
                    user_emb=u_emb_all,
                    item_emb=i_emb_all,
                    adj=adj,
                )
                aux = torch.zeros((), device=self.device)
                for f in bundle.aux_losses:
                    aux = aux + f(ctx)
                if cfg.training.aurl is not None:
                    aux = aux + self._aurl_loss(u_emb_all, i_emb_all, users, pos)
                loss = rec + reg + aux

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                sums["loss"] += float(loss.detach().item())
                sums["rec"] += float(rec.detach().item())
                sums["l2"] += float(reg.detach().item())
                sums["aux"] += float(aux.detach().item())
                n_batches += 1

            for k in sums:
                sums[k] /= max(n_batches, 1)
            line = (
                f"epoch={epoch:>4} time={time.time() - t0:6.1f}s "
                f"loss={sums['loss']:.4f} rec={sums['rec']:.4f} "
                f"aux={sums['aux']:.4f}"
            )

            stop_line = None
            if (epoch + 1) % cfg.training.eval_every == 0 or epoch == 0:
                val_metrics, test_metrics = self._eval(adj)
                line += _format_eval_table(
                    val_metrics, test_metrics, k=max(cfg.eval.k_list)
                )
                with trace_path.open("a") as f:
                    f.write(
                        json.dumps(
                            {"epoch": epoch, "val": val_metrics, "test": test_metrics}
                        )
                        + "\n"
                    )
                out = stopper.observe(epoch, val_metrics, test_metrics)
                if out.improved:
                    self._save_checkpoint(
                        epoch, val_metrics, test_metrics, "best_ckpt.pt"
                    )
                # A new best moves the drift line, so delete band checkpoints that were
                # eligible before but now fall outside it.
                for ep in out.band_drop:
                    (self.out_dir / f"band_ep{ep}.pt").unlink(missing_ok=True)
                if out.band_save:
                    self._save_checkpoint(
                        epoch, val_metrics, test_metrics, f"band_ep{epoch}.pt"
                    )
                line += out.annotation
                stop_line = out.stop

            print(line, flush=True)
            with log_path.open("a") as f:
                f.write(line + "\n")

            if stop_line is not None:
                print(stop_line)
                with log_path.open("a") as f:
                    f.write(stop_line + "\n")
                # Stop epoch loop
                break

        best_metrics = stopper.result()
        (self.out_dir / "metrics.json").write_text(
            json.dumps(best_metrics, indent=2) + "\n"
        )
        return best_metrics

    @torch.no_grad()
    def _eval(self, adj: torch.Tensor) -> tuple[dict[str, float], dict[str, float]]:
        """
        Forward the encoder in eval mode and compute val and test metrics.

        Test ranking masks the full `train.txt` (train plus any carved val). Validation
        ranking masks only the fitted interactions. When no validation split is carved,
        the returned val metrics are empty. When `eval.test` is off, the returned test
        metrics are empty.

        Parameters
        ----------
        adj : torch.Tensor of shape (n_users + n_items, n_users + n_items), float32
            Sparse propagation adjacency to forward through the encoder.

        Returns
        -------
        tuple[dict[str, float], dict[str, float]]
            `(val_metrics, test_metrics)`; `val_metrics` is empty when `val_ratio == 0`,
            `test_metrics` is empty when `eval.test is False`.

        """
        self.bundle.encoder.eval()
        u_emb, i_emb = self.bundle.encoder(adj)
        # SSM head normalises inside the loss only. Ranking stays inner product. This is
        # the Cos-IP form of SSM.
        test_metrics = (
            self.evaluator.evaluate(
                u_emb,
                i_emb,
                self.dataset.test_user_items,
                self.dataset.train_val_user_items,
            )
            if self.cfg.eval.test
            else {}
        )
        val_metrics = (
            self.evaluator.evaluate(
                u_emb,
                i_emb,
                self.dataset.val_user_items,
                self.dataset.user_pos_items,
            )
            if self.dataset.val_user_items
            else {}
        )
        return val_metrics, test_metrics

    def _save_checkpoint(
        self,
        epoch: int,
        val_metrics: dict[str, float],
        test_metrics: dict[str, float],
        filename: str,
    ) -> None:
        """
        Save encoder and metric state to a checkpoint file.

        Model state only: no optimiser or RNG state, so these checkpoints support
        inference and analysis but not resumption of training.

        Parameters
        ----------
        epoch : int
            Epoch at which the checkpoint is taken.
        val_metrics : dict[str, float]
            Validation metrics at this epoch.
        test_metrics : dict[str, float]
            Test metrics at this epoch.
        filename : str
            File to write under the run directory: `"best_ckpt.pt"` for the primary
            metric's best epoch, `"band_ep{epoch}.pt"` for an eligible-window checkpoint
            when `early_stop.save_eligible_checkpoints` is on.

        """
        torch.save(
            {
                "epoch": epoch,
                "encoder_state": self.bundle.encoder.state_dict(),
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            },
            self.out_dir / filename,
        )


def _format_eval_table(
    val_metrics: dict[str, float], test_metrics: dict[str, float], k: int
) -> str:
    """
    Render val/test eval metrics as an aligned, grouped-column block.

    Only splits with a non-empty metrics dict get a row.

    Parameters
    ----------
    val_metrics : dict[str, float]
        Validation metrics for the epoch.
    test_metrics : dict[str, float]
        Test metrics for the epoch.
    k : int
        Which `@k` metrics to show for both splits.

    Returns
    -------
    str
        A multi-line block (header row plus a row per non-empty split). Empty line if no
        splits.

    """
    splits = (("val", val_metrics), ("test", test_metrics))
    labelled = [(lbl, m) for lbl, m in splits if m]
    if not labelled:
        return ""
    ref = labelled[0][1]
    spec = [
        (f"R@{k}", f"recall@{k}"),
        (f"N@{k}", f"ndcg@{k}"),
        (f"PRU@{k}", f"pru@{k}"),
        (f"NOV@{k}", f"novelty@{k}"),
        (f"COV@{k}", f"item_coverage@{k}"),
        ("PRU+", "pru_plus"),
    ]
    cols = [(h, key) for h, key in spec if key in ref]
    val_w = len("test")  # label column width ("val"/"test")
    widths = [max(len(h), len("0.0000")) for h, _ in cols]

    def _row(label: str, metrics: dict[str, float]) -> str:
        cells = [f"{metrics[key]:.4f}".rjust(w) for (_, key), w in zip(cols, widths)]
        return "  " + label.ljust(val_w) + "  " + "  ".join(cells)

    header = (
        "  "
        + " " * val_w
        + "  "
        + "  ".join(h.rjust(w) for (h, _), w in zip(cols, widths))
    )
    rows = [header] + [_row(lbl, m) for lbl, m in labelled]
    return "\n" + "\n".join(rows)
