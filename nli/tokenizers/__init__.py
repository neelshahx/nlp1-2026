from .base import Tokenizer
from .bpe import BPETokenizer
from .wordlevel import WordLevelTokenizer

TOKENIZERS: dict[str, type[Tokenizer]] = {
    WordLevelTokenizer.kind: WordLevelTokenizer,
    BPETokenizer.kind: BPETokenizer,
}


def build_tokenizer(config: dict) -> Tokenizer:
    """``config`` is the ``tokenizer`` block: ``{"kind": ..., **hyperparameters}``."""
    kwargs = {k: v for k, v in config.items() if k != "kind"}
    return TOKENIZERS[config["kind"]](**kwargs)


__all__ = ["Tokenizer", "TOKENIZERS", "build_tokenizer"]
