from ..tokenizers import Tokenizer
from .base import PAIRINGS, Features, Representation, VectorRepresentation, sides
from .bow import BagOfWordsRepresentation
from .indicator import IndicatorRepresentation
from .lengths import LengthsRepresentation
from .ngram import NgramRepresentation

REPRESENTATIONS: dict[str, type[Representation]] = {
    LengthsRepresentation.kind: LengthsRepresentation,
    IndicatorRepresentation.kind: IndicatorRepresentation,
    BagOfWordsRepresentation.kind: BagOfWordsRepresentation,
    NgramRepresentation.kind: NgramRepresentation,
}


def build_representation(config: dict, tokenizer: Tokenizer) -> Representation:
    """``config`` is the ``representation`` block: ``{"kind": ..., **hyperparameters}``."""
    kwargs = {k: v for k, v in config.items() if k != "kind"}
    return REPRESENTATIONS[config["kind"]](tokenizer, **kwargs)


__all__ = ["Features", "Representation", "VectorRepresentation", "PAIRINGS", "sides",
           "REPRESENTATIONS", "build_representation"]
