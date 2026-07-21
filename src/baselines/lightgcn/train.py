"""
Entrypoint for LightGCN.

Usage:
    uv run python -m baselines.lightgcn.train --config configs/lightgcn/gowalla.yaml \
         --out-dir results/lightgcn/seed_2021
    uv run python -m baselines.lightgcn.train --config configs/amssm/gowalla.yaml \
        --out-dir results/amssm/seed_2021
    uv run python -m baselines.lightgcn.train --config configs/aurl/gowalla.yaml \
        --out-dir results/aurl/seed_2021
"""

from baselines.lightgcn import model  # noqa: F401
from common.training.cli import main

if __name__ == "__main__":
    main("lightgcn")
