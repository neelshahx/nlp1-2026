"""Correlated Bayesian t-test with seeds averaged per fold (runs=1) vs the current runs=2, ROPE 0.5 and 1 pt."""
import glob
import baycomp
from nli.store import load_result

def rid(prefix):
    (m,) = glob.glob(f"results/{prefix}*"); return m.split("/", 1)[1]

R = {k: load_result(".", rid(v)) for k, v in {
    "lstm": "snli-glove-bowman-lstm-freeze_embeddings=False", "lstm_fz": "snli-glove-bowman-lstm-freeze_embeddings=True",
    "max": "snli-glove-bowman-max-freeze_embeddings=False", "max_fz": "snli-glove-bowman-max-freeze_embeddings=True",
    "mean": "snli-glove-bowman-mean-freeze_embeddings=False", "mean_fz": "snli-glove-bowman-mean-freeze_embeddings=True",
    "sum": "snli-glove-bowman-sum-freeze_embeddings=False", "sum_fz": "snli-glove-bowman-sum-freeze_embeddings=True",
    "bert": "snli-bert-bowman-combiner=concat",
    "bert_match": "snli-bert-bowman-combiner=matching", "bert_hyp": "snli-bert-bowman-combiner=hypothesis_only",
}.items()}

def per_fold_avg(r):
    df = r.scores[r.scores.metric == "accuracy"]
    return df.groupby("fold").value.mean().sort_index().to_numpy() * 100

def full(r):
    return r.per_fold("accuracy") * 100

pairs = [("lstm", "sum"), ("sum", "sum_fz"), ("lstm", "lstm_fz"), ("max", "max_fz"), ("max", "mean"),
         ("lstm", "max"), ("bert", "mean_fz"),
         ("bert_match", "bert"), ("bert", "bert_hyp"), ("max", "bert_match")]
print(f"{'a vs b':18s} diff | rope  runs=2 (a>b / rope / b>a)   | runs=1 seed-avg (a>b / rope / b>a)")
for a, b in pairs:
    d = full(R[a]).mean() - full(R[b]).mean()
    for rope in (0.5, 1.0):
        p2 = baycomp.two_on_single(full(R[a]), full(R[b]), rope=rope, runs=2)
        p1 = baycomp.two_on_single(per_fold_avg(R[a]), per_fold_avg(R[b]), rope=rope, runs=1)
        f = lambda p: " / ".join(f"{x:.2f}" for x in p)
        print(f"{a+' vs '+b:18s} {d:+.2f} | {rope:.1f}  {f(p2):28s} | {f(p1)}")
