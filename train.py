import argparse
import csv
import os
from datetime import datetime
 
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
 
from tokenizer import BowTokenizer, pair_features
from mlp import build_model
 
# Settings (defaults can be overridden on the command line)
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="mlp", choices=["mlp", "deep_mlp", "res_mlp"])
parser.add_argument("--n_blocks", type=int, default=4, help="depth for deep_mlp / res_mlp")
parser.add_argument("--hidden_dim", type=int, default=512)
parser.add_argument("--max_vocab", type=int, default=20000)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
 
MODEL = args.model
N_BLOCKS = args.n_blocks if MODEL != "mlp" else "n/a"   # single-layer MLP has no blocks
MAX_VOCAB = args.max_vocab
HIDDEN_DIM = args.hidden_dim
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LR = args.lr
SEED = args.seed
RESULTS_DIR = "results"
 
torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(RESULTS_DIR, exist_ok=True)
RUN_NAME = f"{MODEL}_{datetime.now():%Y%m%d-%H%M%S}"
LOG_PATH = os.path.join(RESULTS_DIR, f"{RUN_NAME}.txt")
 
 
def log(message):
    print(message)
    with open(LOG_PATH, "a") as f:
        f.write(message + "\n")
 
 
def featurize(tok, split):
    u = tok.transform(list(split["premise"]))
    v = tok.transform(list(split["hypothesis"]))
    return pair_features(u, v), np.array(split["label"])
 
 
def batches(X, y, batch_size, shuffle=False):
    order = np.random.permutation(X.shape[0]) if shuffle else np.arange(X.shape[0])
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        xb = torch.from_numpy(X[idx].toarray()).to(device)
        yb = torch.from_numpy(y[idx]).long().to(device)
        yield xb, yb
 
 
def evaluate(model, X, y):
    model.eval()                      # turns dropout off
    correct = 0
    with torch.no_grad():
        for xb, yb in batches(X, y, 1024):
            correct += (model(xb).argmax(dim=1) == yb).sum().item()
    return correct / len(y)
 
 
def main():
    log(f"Run: {RUN_NAME}")
    log(f"Settings: model={MODEL}, n_blocks={N_BLOCKS}, max_vocab={MAX_VOCAB}, "
        f"hidden_dim={HIDDEN_DIM}, batch_size={BATCH_SIZE}, epochs={EPOCHS}, "
        f"lr={LR}, seed={SEED}, device={device}")
 
    # 1. Load data, dropping pairs without a gold label (-1)
    ds = load_dataset("stanfordnlp/snli")
    ds = ds.filter(lambda ex: ex["label"] != -1)
    train, val, test = ds["train"], ds["validation"], ds["test"]
 
    # 2. Tokenizer: build the vocabulary on training sentences only
    tok = BowTokenizer(min_freq=2, max_size=MAX_VOCAB)
    tok.fit(list(train["premise"]) + list(train["hypothesis"]))
    log(f"Vocabulary size: {len(tok)}")
 
    X_train, y_train = featurize(tok, train)
    X_val, y_val = featurize(tok, val)
    X_test, y_test = featurize(tok, test)
 
    # 3. Model, loss, optimizer
    model = build_model(MODEL, X_train.shape[1], HIDDEN_DIM, N_BLOCKS).to(device)
    log(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
 
    # 4. Train, keeping the weights with the best validation accuracy
    best_acc, best_epoch, best_state = 0.0, 0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()                 # turns dropout on
        total_loss, correct = 0.0, 0
        for xb, yb in batches(X_train, y_train, BATCH_SIZE, shuffle=True):
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
 
        train_acc = correct / len(y_train)
        val_acc = evaluate(model, X_val, y_val)
        log(f"Epoch {epoch}: train loss {total_loss / len(y_train):.4f}, "
            f"train acc {train_acc:.3f}, val acc {val_acc:.3f}")
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
 
    # 5. Final evaluation on the test set with the best model
    model.load_state_dict(best_state)
    test_acc = evaluate(model, X_test, y_test)
    log(f"Best val acc: {best_acc:.3f} (epoch {best_epoch})")
    log(f"Test acc: {test_acc:.3f}")
 
    # 6. One row per run in results/summary.csv, for the comparison table
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    new_file = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["run", "model", "n_blocks", "hidden_dim", "max_vocab",
                             "epochs", "lr", "best_epoch", "val_acc", "test_acc"])
        writer.writerow([RUN_NAME, MODEL, N_BLOCKS, HIDDEN_DIM, MAX_VOCAB,
                         EPOCHS, LR, best_epoch, f"{best_acc:.4f}", f"{test_acc:.4f}"])
    log(f"Saved log to {LOG_PATH} and results to {summary_path}")
 
 
if __name__ == "__main__":
    main()