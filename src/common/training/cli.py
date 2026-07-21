"""Generic entrypoint for model training."""

import argparse
from pathlib import Path

import torch

from .config import Config, apply_overrides, load_config
from .data import InteractionDataset
from .evaluator import Evaluator
from .registry import build_model
from .seed import set_global_seed
from .trainer import Trainer


def main(model_name: str) -> dict[str, float]:
    """
    Parse args, build dataset+model, train.

    Parameters
    ----------
    model_name : str
        Name of the model to train; must match the `name` field of the loaded config and
        selects the registered build-fn.

    Returns
    -------
    dict[str, float]
        Best metrics returned by the trainer.

    """
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help=(
            "dotted-path config override(s), repeatable. The path is matched "
            "against the `Config` dataclass. Top-level keys take no section prefix; "
            "nested keys take their section. Unknown keys are rejected."
        ),
    )
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    if cfg.name != model_name:
        raise ValueError(
            f"config name={cfg.name!r} but train.py invoked for {model_name!r}"
        )

    if cfg.training.reg_form not in ("yu2022", "he2020"):
        raise ValueError(
            f"training.reg_form must be 'yu2022' or 'he2020', "
            f"got {cfg.training.reg_form!r}."
        )

    _validate_early_stop(cfg)

    set_global_seed(cfg.seed, deterministic=True)

    device = cfg.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"config requests device={cfg.device!r} but torch.cuda.is_available() "
            "is False; refusing to silently fall back to CPU."
        )

    datasets_home = Path(__import__("os").environ.get("DATASETS_HOME", "datasets"))
    split_dir = datasets_home / cfg.dataset / cfg.split_path
    val = cfg.training.val
    dataset = InteractionDataset.from_dir(
        split_dir,
        val_ratio=val.ratio if val is not None else 0.0,
        val_seed=val.seed if val is not None else None,
        val_carve=val.carve if val is not None else "temporal",
    )
    n_val = sum(len(v) for v in dataset.val_user_items.values())
    print(
        f"dataset={cfg.dataset} split_dir={split_dir} users={dataset.n_users} "
        f"items={dataset.n_items} train={dataset.n_train} val={n_val} "
        f"test={len(dataset.test_edges)} "
        f"early_stop_split={cfg.training.early_stop.split}"
    )

    bundle = build_model(model_name, cfg, dataset, device)
    evaluator = Evaluator(dataset, cfg.eval)
    trainer = Trainer(bundle, dataset, evaluator, cfg, device, args.out_dir)
    return trainer.train()


def _validate_early_stop(cfg: Config) -> None:
    """
    Validate the `training.early_stop` block against the eval config.

    Parameters
    ----------
    cfg : Config
        Run configuration, after overrides.

    """
    early_stop = cfg.training.early_stop

    val = cfg.training.val
    val_active = val is not None and val.ratio > 0.0
    if early_stop.split == "val" and not val_active:
        raise ValueError(
            "training.early_stop.split='val' requires a training.val block with "
            "ratio > 0; no validation split is carved when training.val is null or "
            "absent. Add `val: {}` under `training:` to carve with the defaults."
        )
    if early_stop.split == "test" and not cfg.eval.test:
        raise ValueError(
            "training.early_stop.split='test' requires eval.test=true; test metrics "
            "are not computed otherwise."
        )

    k_list = tuple(cfg.eval.k_list)
    available = _evaluator_metric_keys(k_list, cfg.eval.compute_pru)

    checks = [("mode", early_stop.mode, "metric", early_stop.metric)]
    if early_stop.secondary_metric is not None:
        checks.append(
            (
                "secondary_mode",
                early_stop.secondary_mode,
                "secondary_metric",
                early_stop.secondary_metric,
            )
        )

    for mode_field, mode, metric_field, metric in checks:
        if mode not in ("max", "min"):
            raise ValueError(
                f"training.early_stop.{mode_field} must be 'max' or 'min', got {mode!r}"
            )
        if metric not in available:
            raise ValueError(
                f"training.early_stop.{metric_field}={metric!r} is not produced by "
                f"the evaluator under this eval config (k_list={k_list}, "
                f"compute_pru={cfg.eval.compute_pru}). Available: {sorted(available)}"
            )


def _evaluator_metric_keys(k_list: tuple[int, ...], compute_pru: bool) -> set[str]:
    """
    Scalar metric keys the `Evaluator` emits for a given eval config.

    Parameters
    ----------
    k_list : tuple[int, ...]
        The evaluator's cut-offs.
    compute_pru : bool
        Whether the PRU family is computed.

    Returns
    -------
    set[str]
        Every scalar key the evaluator's output dict will contain.

    """
    keys: set[str] = set()
    for k in k_list:
        for band in ("", "_head", "_tail"):
            keys.add(f"recall@{k}{band}")
            keys.add(f"ndcg@{k}{band}")
        keys.add(f"novelty@{k}")
        keys.add(f"item_coverage@{k}")
        keys.add(f"eff_coverage@{k}")
        if compute_pru:
            keys.add(f"pru@{k}")
    if compute_pru:
        keys.add("pru_plus")
    return keys
