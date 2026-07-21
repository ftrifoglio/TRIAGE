"""Model registry."""

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from .config import Config
    from .data import InteractionDataset
    from .sampler import NeighbourhoodSampler
    from .trainer import AuxLoss


@dataclass
class ModelBundle:
    """
    Everything `Trainer` needs to run a model.

    Attributes
    ----------
    encoder : nn.Module
        The encoder.
    neighbourhood_sampler : NeighbourhoodSampler
        Sampler supplying the propagation adjacency each epoch.
    aux_losses : list[AuxLoss]
        Auxiliary loss terms added to the per-batch objective.

    """

    encoder: nn.Module
    neighbourhood_sampler: "NeighbourhoodSampler"
    aux_losses: list["AuxLoss"] = field(default_factory=list)


BuildFn = Callable[["Config", "InteractionDataset", str], ModelBundle]
_REGISTRY: dict[str, BuildFn] = {}


def register_model(name: str) -> Callable[[BuildFn], BuildFn]:
    """
    Register a baseline by name.

    Parameters
    ----------
    name : str
        Registry key the build function is registered under.

    Returns
    -------
    Callable[[BuildFn], BuildFn]
        A decorator that records its target build function and returns it unchanged.

    """

    def deco(fn: BuildFn) -> BuildFn:
        """
        Register `fn` under the enclosing `name` and return it unchanged.

        Parameters
        ----------
        fn : BuildFn
            The build function to register.

        Returns
        -------
        BuildFn
            The same function `fn`, unmodified.

        """
        if name in _REGISTRY:
            raise ValueError(f"model {name!r} already registered")
        _REGISTRY[name] = fn
        return fn

    return deco


def build_model(
    name: str, cfg: "Config", dataset: "InteractionDataset", device: str
) -> ModelBundle:
    """
    Construct a model bundle by registry name.

    Parameters
    ----------
    name : str
        Registry key of the baseline to build.
    cfg : Config
        Run configuration passed to the build function.
    dataset : InteractionDataset
        Dataset passed to the build function.
    device : str
        Target device passed to the build function.

    Returns
    -------
    ModelBundle
        The bundle produced by the registered build function for `name`.

    """
    if name not in _REGISTRY:
        raise KeyError(f"model {name!r} not registered. Known: {list(_REGISTRY)}")
    bundle = _REGISTRY[name](cfg, dataset, device)
    if cfg.encoder.eps != 0.0 and not hasattr(bundle.encoder, "eps"):
        warnings.warn(
            f"encoder.eps={cfg.encoder.eps} is ignored by "
            f"{type(bundle.encoder).__name__} (model {name!r}); overriding to 0.0.",
            UserWarning,
            stacklevel=2,
        )
        cfg.encoder.eps = 0.0
    return bundle
