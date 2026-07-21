"""
Apply popularity compensation (PC) to a trained run and select one checkpoint.

The PC config holds the per-dataset PC settings (`eval.pc_a`, `eval.pc_b`) and the
`dataset` they were tuned on. The source run is `--run-dir`. Its `dataset` must match
the PC config's, and every other setting comes from its `config.json`.

The candidates are the eligible-window checkpoints (`band_ep*.pt`) in the source run,
which it writes when `training.early_stop.save_eligible_checkpoints` is on. Each
candidate is scored on the val split with PC applied at the fixed `pc_a` and `pc_b`.

Selection uses the source run's early-stop settings, applied to the PC-scored val
metrics. After scoring all candidates, the best PC-scored primary metric sets an
internal peak. Any other candidate whose PC-scored primary is within `drift_tolerance`
from the peak is admissible. The candidate that sets the peak is always admissible. The
winner is the admissible candidate with the best PC-scored secondary metric.

Only the winner is scored on the test split. Its val and test metrics are written to
`metrics.json` in `--out-dir`.

Usage:
    uv run python scripts/pc.py --config configs/pc/gowalla.yaml \
        --run-dir results/lightgcn/gowalla/seed_2021 \
        --out-dir results/pc/gowalla/seed_2021
"""

import argparse
import functools
import importlib
import json
from importlib.util import find_spec
from pathlib import Path

import torch
import yaml

from baselines.popularity_compensation import popularity_compensation
from common.training.config import load_config
from common.training.data import InteractionDataset
from common.training.evaluator import Evaluator
from common.training.registry import build_model
from common.training.seed import set_global_seed


def main() -> None:
    """Score the window at frozen PC settings and select one checkpoint."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    pc_cfg = yaml.safe_load(args.config.read_text())
    pc_a = pc_cfg["eval"]["pc_a"]
    pc_b = pc_cfg["eval"]["pc_b"]
    adjust = functools.partial(popularity_compensation, pc_a=pc_a, pc_b=pc_b)
    run_dir = args.run_dir
    out_dir = args.out_dir

    cfg = load_config(run_dir / "config.json")
    if cfg.dataset != pc_cfg["dataset"]:
        raise SystemExit(
            f"PC config is for dataset={pc_cfg['dataset']!r} but {run_dir} was trained "
            f"on dataset={cfg.dataset!r}"
        )
    split = cfg.training.early_stop.split
    metric = cfg.training.early_stop.metric
    mode = cfg.training.early_stop.mode
    secondary_metric = cfg.training.early_stop.secondary_metric
    assert secondary_metric is not None
    secondary_mode = cfg.training.early_stop.secondary_mode
    drift_tolerance = cfg.training.early_stop.drift_tolerance
    if split != "val":
        raise SystemExit(f"selection scores the val split; config has split={split!r}")

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
    if val is None:
        raise SystemExit("run was trained without a validation carve; cannot select")
    dataset = InteractionDataset.from_dir(
        split_dir,
        val_ratio=val.ratio,
        val_seed=val.seed,
        val_carve=val.carve,
    )
    module = cfg.name if find_spec(cfg.name) else f"baselines.{cfg.name}"
    importlib.import_module(f"{module}.model")
    bundle = build_model(cfg.name, cfg, dataset, device)
    evaluator = Evaluator(dataset, cfg.eval)
    band = sorted(
        run_dir.glob("band_ep*.pt"),
        key=lambda q: int(q.stem.removeprefix("band_ep")),
    )
    if not band:
        raise SystemExit(
            f"no band_ep*.pt in {run_dir}; re-run it with "
            "training.early_stop.save_eligible_checkpoints=true"
        )
    print(
        f"PC: {cfg.name}/{cfg.dataset} seed {cfg.seed}, {len(band)} checkpoints",
        flush=True,
    )

    embeddings: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    scored = []
    for path in band:
        epoch = int(path.stem.removeprefix("band_ep"))
        ck = torch.load(path, map_location=device, weights_only=False)
        bundle.encoder.load_state_dict(ck["encoder_state"])
        bundle.encoder.to(device)
        bundle.encoder.eval()
        adj = bundle.neighbourhood_sampler.sample(0, {"encoder": bundle.encoder}).to(
            device
        )
        with torch.no_grad():
            user_emb, item_emb = bundle.encoder(adj)
        embeddings[epoch] = (user_emb.cpu(), item_emb.cpu())
        val_m = evaluator.evaluate(
            user_emb,
            item_emb,
            dataset.val_user_items,
            dataset.user_pos_items,
            score_adjust=adjust,
        )
        scored.append({"epoch": epoch, **{f"val_{k}": v for k, v in val_m.items()}})
        print(
            f"  ep{epoch:5d}  val {metric}={val_m[metric]:.5f} "
            f"{secondary_metric}={val_m[secondary_metric]:.5f}",
            flush=True,
        )

    pick = max if mode == "max" else min
    best_metric = pick(c[f"{split}_{metric}"] for c in scored)
    selections = {}

    if mode == "min":
        line = best_metric * (1.0 + drift_tolerance)
        admissible = [c for c in scored if c[f"{split}_{metric}"] <= line]
    else:
        line = best_metric * (1.0 - drift_tolerance)
        admissible = [c for c in scored if c[f"{split}_{metric}"] >= line]

    pick_secondary = max if secondary_mode == "max" else min
    win = pick_secondary(admissible, key=lambda c: c[f"{split}_{secondary_metric}"])

    user_emb, item_emb = (t.to(device) for t in embeddings[int(win["epoch"])])
    test_m = evaluator.evaluate(
        user_emb,
        item_emb,
        dataset.test_user_items,
        dataset.train_val_user_items,
        score_adjust=adjust,
    )
    print(
        f"  delta={drift_tolerance}: ep {win['epoch']} "
        f"val {secondary_metric}={win[f'{split}_{secondary_metric}']:.5f} "
        f"test {secondary_metric}={test_m[secondary_metric]:.5f}",
        flush=True,
    )

    selections[f"{drift_tolerance}"] = {
        "epoch": win["epoch"],
        **{k: v for k, v in win.items() if k != "epoch"},
        **{f"test_{k}": v for k, v in test_m.items()},
    }

    source_peak_epoch = json.load((run_dir / "metrics.json").open())["epoch"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "pc_a": pc_a,
                "pc_b": pc_b,
                "seed": cfg.seed,
                "source_run": str(run_dir),
                "source_run_peak_epoch": source_peak_epoch,
                "selections": selections,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {out_dir}/metrics.json", flush=True)


if __name__ == "__main__":
    main()
