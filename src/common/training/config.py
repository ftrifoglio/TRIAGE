"""All configurations."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EncoderCfg:
    """
    Encoder configurations.

    Attributes
    ----------
    emb_dim : int
        Embedding dimensionality.
    n_layers : int
        Number of propagation layers.
    readout_ego : bool
        Whether the layer-mean readout includes the layer-0 table. `True` (default)
        averages layers `0..K`, `False` layers `1..K`.
    eps : float
        The strength of the sign-aligned, L2-normalised uniform perturbation `eps *
        sign(e) * U(0,1)` injected after each propagation step when the encoder is asked
        for a perturbed view.

    """

    emb_dim: int = 64
    n_layers: int = 3
    readout_ego: bool = True
    eps: float = 0.0


@dataclass
class ValCfg:
    """
    Validation-carve configurations.

    Attributes
    ----------
    ratio : float
        Fraction of each user's training items carved into the validation split.
    seed : int
        Seed for the validation carve when `carve == "random"`. "temporal" ignores it.
    carve : str
        Strategy for selecting each user's validation items from `train.txt`:
        `"temporal"` (default) holds out the most-recent tail of the time-ordered
        interactions. `"random"` holds out a uniform random subset.

    """

    ratio: float = 0.125
    seed: int = 2020
    carve: str = "temporal"


@dataclass
class EarlyStopCfg:
    """
    Early-stopping configurations.

    Attributes
    ----------
    patience : int
        Number of evaluations without improvement on `metric` before training stops.
    split : str
        Split whose `metric` is monitored for early stopping.
    metric : str
        Key of the primary metric monitored for early stopping.
    mode : str
        Optimisation direction for `metric`: one of `"max"` or `"min"`.
    secondary_metric : str | None
        Key of a second metric monitored after `metric` has plateaued.
    secondary_mode : str
        Optimisation direction for `secondary_metric`.
    secondary_patience : int | None
        Evaluations without improvement in `secondary_metric` before stage two stops.
        `None` (default) reuses `patience`. Tracked from the arming point, not from the
        start of the run.
    drift_tolerance : float
        Stage-two guard on the primary metric: training stops as soon as `metric` falls
        more than this fraction below its best value.
    admissibility_floor : float | None
        Two-stage only (ignored when `secondary_metric` is `None`). When stage one
        plateaus, abandon the run instead of arming stage two if the run's best `metric`
        is strictly worse than this value; `metrics.json` then records `abandoned=True`.
        Intended for sweeps, to abort trials whose best cannot fall within
        `drift_tolerance` of the best across trials so far (for `mode="max"`, set it to
        that best * (1.0 - drift_tolerance)). `None` (default) never abandons and
        reproduces the two-stage rule exactly.
    save_eligible_checkpoints : bool
        Write every evaluated checkpoint whose `metric` lies within `drift_tolerance` of
        the run's best so far to `band_ep{epoch}.pt`, and delete band files that drop
        out of the window when the best improves, so the run directory only ever holds
        the current window. `True` (default) writes both `band_ep{epoch}.pt` and
        `best_ckpt.pt`.

    """

    patience: int = 10
    split: str = "val"
    metric: str = "recall@20"
    mode: str = "max"
    secondary_metric: str | None = "recall@20_tail"
    secondary_mode: str = "max"
    secondary_patience: int | None = None
    drift_tolerance: float = 0.02
    admissibility_floor: float | None = None
    save_eligible_checkpoints: bool = True


@dataclass
class SSMCfg:
    """
    SSM configurations.

    Attributes
    ----------
    tau : float
        Softmax temperature.
    margin : float
        Constant additive cosine margin `m` (similarity units) subtracted from the
        positive logit.
    beta : float
        Negative-sampling distribution exponent. Each batch draws `n_neg` shared
        negatives from `p_n(i) proportional to deg_i^beta`. `1.0` (default) is
        equivalent to in-batch negative sampling in expectation. `0.0` is uniform
        sampling.
    n_neg : int
        Number `N` of shared negatives drawn per batch. `0` (default) resolves to
        `training.batch_size`.

    """

    tau: float = 0.1
    margin: float = 0.0
    beta: float = 1.0
    n_neg: int = 0


@dataclass
class AURLCfg:
    """
    AURL configurations.

    Attributes
    ----------
    align_weight : float
        Weight of the mean of the user-side and item-side popular-vs-unpopular
        multi-kernel MMD.
    kernel_mul : float
        Geometric ratio between consecutive RBF kernel bandwidths.
    kernel_num : int
        Number of RBF kernels in the multi-kernel MMD estimator.
    mmd_batch : int
        Maximum number of entities sampled per group per side at each step.
    uniform_weight : float
        Weight of the mean of the user-side and item-side Wang & Isola uniformity loss.
    t : float
        Gaussian-potential temperature. `2.0` (default) follows Wang & Isola (2020)
        whose reference implementation defaults to t = 2 (their Fig. 6).
    uniform_batch : int
        Maximum number of entities sampled per side at each step.

    """

    align_weight: float = 1.0
    kernel_mul: float = 2.0
    kernel_num: int = 5
    mmd_batch: int = 1024
    uniform_weight: float = 1.0
    t: float = 2.0
    uniform_batch: int = 1024


@dataclass
class TrainingCfg:
    """
    Training configurations.

    Attributes
    ----------
    lr : float
        Adam learning rate.
    batch_size : int
        BPR sampler batch size.
    l2 : float
        Weight on the L2 regularisation term.
    epochs : int
        Maximum number of training epochs.
    early_stop : EarlyStopCfg
        Early-stopping configurations.
    eval_every : int
        Run evaluation every this many epochs.
    val : ValCfg | None
        Validation-carve configurations.
    reg_form : str
        L2 regulariser functional form. `"yu2022"` (default) is the SELFRec unsquared
        Frobenius term on the propagated user+pos embeddings. `"he2020"` is the original
        LightGCN squared-L2 term on the ego (layer-0) user+pos+neg embeddings.
    bpr_sampling : str
        BPR triple sampling convention. `"per_edge"` (default) permutes the training
        edges and visits each once per epoch. `"per_user"` mirrors the canonical
        LightGCN sampler: each batch draws `batch_size` users uniformly and one uniform
        positive per user.
    ssm : SSMCfg | None
        SSM configurations.
    aurl : AURLCfg | None
        AURL configurations.

    """

    lr: float = 1.0e-3
    batch_size: int = 2048
    l2: float = 1.0e-4
    epochs: int = 1000
    early_stop: EarlyStopCfg = field(default_factory=EarlyStopCfg)
    eval_every: int = 5
    val: ValCfg | None = None
    reg_form: str = "yu2022"
    bpr_sampling: str = "per_edge"
    ssm: SSMCfg | None = None
    aurl: AURLCfg | None = None


@dataclass
class EvalCfg:
    """
    Evaluation configurations.

    Attributes
    ----------
    k_list : tuple[int, ...]
        K's of the @K metrics.
    user_chunk : int
        Number of test users scored per chunk.
    head_quantile : float
        Degree quantile separating head from tail items. `0.8` means the top 20% by
        degree counts as "head".
    compute_pru : bool
        Whether to compute the PRU metrics at every eval.
    test : bool
        Whether to rank the test split and compute test metrics at every eval.

    """

    k_list: tuple[int, ...] = (10, 20)
    user_chunk: int = 1024
    head_quantile: float = 0.8
    compute_pru: bool = True
    test: bool = False


@dataclass
class Config:
    """
    Full run configuration.

    Attributes
    ----------
    name : str
        Model name. Must match the model the run is invoked for.
    dataset : str
        Dataset directory name under the datasets home.
    split_path : str
        Subdirectory under `dataset` holding the `train.txt`/`test.txt` to load.
        Defaults to `"triage_splits"`.
    seed : int
        Global random seed. Defaults to 2020.
    device : str
        Requested torch device. Defaults to `"cuda"`.
    encoder : EncoderCfg
        Encoder configurations.
    training : TrainingCfg
        Training configurations.
    eval : EvalCfg
        Evaluation configurations.
    model : dict[str, Any]
        Model-specific hyperparameters passed through to the build-fn.

    """

    name: str
    dataset: str
    split_path: str = "triage_splits"
    seed: int = 2020
    device: str = "cuda"
    encoder: EncoderCfg = field(default_factory=EncoderCfg)
    training: TrainingCfg = field(default_factory=TrainingCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    model: dict[str, Any] = field(default_factory=dict)


def load_config(path: Path) -> Config:
    """
    Load YAML config and validate against dataclasses.

    Parameters
    ----------
    path : Path
        Path to the YAML config file.

    Returns
    -------
    Config
        Parsed configuration with nested encoder/training/eval sections populated from
        the YAML.

    """
    raw = yaml.safe_load(Path(path).read_text())
    encoder = EncoderCfg(**raw.get("encoder", {}))
    raw_training = raw.get("training", {})
    nested = ("val", "early_stop", "ssm", "aurl")
    training = TrainingCfg(**{k: v for k, v in raw_training.items() if k not in nested})
    if "early_stop" in raw_training:
        training.early_stop = EarlyStopCfg(**raw_training["early_stop"])
    if "val" in raw_training:
        v = raw_training["val"]
        training.val = ValCfg(**v) if v is not None else None
    if "ssm" in raw_training:
        s = raw_training["ssm"]
        training.ssm = SSMCfg(**s) if s is not None else None
    if "aurl" in raw_training:
        r = raw_training["aurl"]
        training.aurl = AURLCfg(**r) if r is not None else None
    eval_cfg = EvalCfg(**raw.get("eval", {}))
    if "k_list" in raw.get("eval", {}):
        eval_cfg.k_list = tuple(raw["eval"]["k_list"])
    optional = {
        k: raw[k] for k in ("split_path", "seed", "device", "model") if k in raw
    }
    return Config(
        name=raw["name"],
        dataset=raw["dataset"],
        encoder=encoder,
        training=training,
        eval=eval_cfg,
        **optional,
    )


def apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    """
    Apply dotted-path overrides like `model.tau=0.1` or `training.lr=5e-4`.

    Parameters
    ----------
    cfg : Config
        Configuration to mutate in place.
    overrides : list[str]
        Dotted-path `key=value` strings; the value is parsed as a Python literal,
        falling back to the raw string if parsing fails.

    Returns
    -------
    Config
        The same `cfg` instance with the overrides applied.

    """
    import ast
    from dataclasses import is_dataclass
    from typing import get_args, get_type_hints

    for ov in overrides:
        key, _, val = ov.partition("=")
        try:
            parsed = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            parsed = val
        parts = key.split(".")
        obj: Any = cfg
        for p in parts[:-1]:
            if isinstance(obj, dict):
                obj = obj[p]
                continue
            nxt = getattr(obj, p)
            if nxt is None:
                # If nxt is an optional Cfg sub-section instantiate its default block so
                # the override can reach into it
                hint = get_type_hints(type(obj)).get(p)
                inner = [a for a in get_args(hint) if a is not type(None)]
                if (
                    len(inner) == 1
                    and isinstance(inner[0], type)
                    and is_dataclass(inner[0])
                ):
                    nxt = inner[0]()
                    setattr(obj, p, nxt)
            obj = nxt
        last = parts[-1]
        if isinstance(obj, dict):
            obj[last] = parsed
        else:
            if not hasattr(obj, last):
                raise ValueError(
                    f"override {key!r} targets unknown config field {last!r}; "
                    f"check the section prefix (e.g. split_path is top-level, "
                    f"not under training)"
                )
            setattr(obj, last, parsed)
    return cfg
