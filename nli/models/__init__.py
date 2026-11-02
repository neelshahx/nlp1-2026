from .base import Model
from .loglinear import LogLinearModel
from .majority import MajorityModel
from .mlp import MLPModel

MODELS: dict[str, type[Model]] = {
    MajorityModel.kind: MajorityModel,
    LogLinearModel.kind: LogLinearModel,
    MLPModel.kind: MLPModel,
}


def build_model(config: dict, seed: int) -> Model:
    """``config`` is the ``model`` block: ``{"kind": ..., **hyperparameters}``."""
    kwargs = {k: v for k, v in config.items() if k != "kind"}
    return MODELS[config["kind"]](seed=seed, **kwargs)


__all__ = ["Model", "MODELS", "build_model"]
