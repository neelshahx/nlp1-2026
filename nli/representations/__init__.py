from ..tokenizers import Tokenizer
from .base import PAIRINGS, Features, Representation, VectorRepresentation, sides
from .bert import BertRepresentation
from .bow import BagOfWordsRepresentation
from .glove import GloveRepresentation, SequencePairs
from .lengths import LengthsRepresentation
from .ngram import NgramRepresentation

REPRESENTATIONS: dict[str, type[Representation]] = {
    LengthsRepresentation.kind: LengthsRepresentation,
    BagOfWordsRepresentation.kind: BagOfWordsRepresentation,
    NgramRepresentation.kind: NgramRepresentation,
    BertRepresentation.kind: BertRepresentation,
    GloveRepresentation.kind: GloveRepresentation,
}


def build_representation(config: dict, tokenizer: Tokenizer) -> Representation:
    """``config`` is the ``representation`` block: ``{"kind": ..., **hyperparameters}``."""
    kwargs = {k: v for k, v in config.items() if k != "kind"}
    return REPRESENTATIONS[config["kind"]](tokenizer, **kwargs)


__all__ = [
    "Features",
    "SequencePairs",
    "Representation",
    "VectorRepresentation",
    "PAIRINGS",
    "sides",
    "REPRESENTATIONS",
    "build_representation",
]
