"""Local sanity check for Bowman encoders on a slice of real SNLI before a Snellius run (shell).

Fits each encoder on the same random train sample with the same seed, then prints per-epoch history, dev accuracy,
and a save/load round trip. Expect falling train loss, val accuracy well above 0.33, and encoders within a few points.

    uv run python scripts/smoke_bowman.py                          # lstm vs gru, 20k train pairs, 3 epochs
    uv run python scripts/smoke_bowman.py --encoders gru --n-train 5000 --epochs 2
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nli.data import load_snli  # noqa: E402
from nli.models.bowman import BowmanModel  # noqa: E402
from nli.representations.glove import GloveRepresentation  # noqa: E402
from nli.tokenizers.treebank import TreebankTokenizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoders", nargs="+", default=["lstm", "gru"])
    parser.add_argument("--n-train", type=int, default=20_000)
    parser.add_argument("--n-dev", type=int, default=2_000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto = cuda, else mps, else cpu")
    args = parser.parse_args()

    snli = load_snli()
    train = snli["train"].sample(n=args.n_train, random_state=args.seed).reset_index(drop=True)
    dev = snli["dev"].sample(n=args.n_dev, random_state=args.seed).reset_index(drop=True)

    t0 = time.monotonic()
    rep = GloveRepresentation(TreebankTokenizer()).fit(train)  # vocab from the train sample only, as in a fold
    X_train, y_train = rep.transform(train), train["label"].to_numpy()
    X_dev, y_dev = rep.transform(dev), dev["label"].to_numpy()
    print(f"glove: {rep.stats}, dev unk rate {rep.unk_rate(dev):.3f}, {time.monotonic() - t0:.0f}s")

    for encoder in args.encoders:
        model = BowmanModel(seed=args.seed, encoder=encoder, num_epochs=args.epochs, device=args.device)
        t0 = time.monotonic()
        model.fit(X_train, y_train)  # prints one history line per epoch
        probs = model.predict_proba(X_dev)
        assert probs.shape == (len(dev), 3) and np.allclose(probs.sum(axis=1), 1)
        dev_acc = float((probs.argmax(axis=1) == y_dev).mean())

        with tempfile.TemporaryDirectory() as d:
            model.save(Path(d) / "m")
            reloaded = BowmanModel.load(Path(d) / "m").predict_proba(X_dev)
        roundtrip = np.allclose(reloaded, probs, atol=1e-5)

        losses = [e["train_loss"] for e in model.history]
        print(
            f"== {encoder}: dev acc {dev_acc:.3f} | best val acc {max(e['val_accuracy'] for e in model.history):.3f}"
            f" | train loss {' -> '.join(f'{x:.3f}' for x in losses)}"
            f" | {time.monotonic() - t0:.0f}s | save/load ok: {roundtrip}"
        )


if __name__ == "__main__":
    main()
