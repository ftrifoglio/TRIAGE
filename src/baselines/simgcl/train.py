"""
Entrypoint for SimGCL.

Usage:
    uv run python -m baselines.simgcl.train --config configs/simgcl/gowalla.yaml \
         --out-dir results/simgcl/seed_2021
"""

from baselines.simgcl import model  # noqa: F401
from common.training.cli import main

if __name__ == "__main__":
    main("simgcl")
