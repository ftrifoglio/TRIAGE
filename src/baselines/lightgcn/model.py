"""LightGCN bundle: encoder + identity sampler + no aux losses."""

from common.training.config import Config
from common.training.data import InteractionDataset, build_norm_adj
from common.training.encoder import LightGCNEncoder
from common.training.registry import ModelBundle, register_model
from common.training.sampler import IdentitySampler


@register_model("lightgcn")
def build(cfg: Config, dataset: InteractionDataset, device: str) -> ModelBundle:
    """
    Build the LightGCN bundle.

    Parameters
    ----------
    cfg : Config
        Full run configuration.
    dataset : InteractionDataset
        Train graph and index sizes used to size the encoder and build the normalised
        adjacency.
    device : str
        Torch device onto which the normalised adjacency is moved.

    Returns
    -------
    ModelBundle
        Bundle of the LightGCN encoder, an `IdentitySampler` over the normalised
        adjacency, and an empty auxiliary-loss list.

    """
    encoder = LightGCNEncoder(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        emb_dim=cfg.encoder.emb_dim,
        n_layers=cfg.encoder.n_layers,
        readout_ego=cfg.encoder.readout_ego,
    )
    norm_adj = build_norm_adj(dataset).to(device)
    return ModelBundle(
        encoder=encoder,
        neighbourhood_sampler=IdentitySampler(norm_adj),
        aux_losses=[],
    )
