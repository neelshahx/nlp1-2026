# Project 2: draft 1 (slide script for Keynote)

Assumed known: NLI and the SNLI labels, embeddings, RNNs, accuracy and CV, the tutorial pipeline, our Project 1 slides.
Explained in the deck: Bowman architecture, protocol (5 folds × 2 seeds, same splits for every run), annotation artifacts (mentioned briefly).

---

## Slide 1: Two embeddings for SNLI. Does each tokenizer fit?

**Bullets** (full text, for us)
1. **GloVe 6B-300d** (Wikipedia + Gigaword, 400k-word vocabulary): the same family the SNLI authors used. It's our main embedding, and every Bowman result in this deck uses it.
2. **Frozen bert-base-uncased** (books + Wikipedia, 30,522 WordPiece units) as the contextual alternative. It was never trained on NLI labels, so no labels leak into our CV.
3. **We matched the tokenizer to each model.** GloVe gets treebank word tokens, the units its vectors were trained on; BERT gets its own WordPiece. The cost is that tokenizer and embedding always change together, so any GloVe–BERT difference belongs to the pair, not to the embedding alone.
4. **Coverage was a non-issue for both models.** GloVe lacks 23% of distinct SNLI words but only 0.24% of tokens; BERT splits 59% of distinct words, yet these are 2.8% of tokens and none become [UNK]. The gaps are rare words: hyphenated compounds (*blond-hair*), typos, words seen once. Both vocabularies cover nearly every token the models see.
5. **Compared with Project 1 (BPE-8k bag of words):** words are now pretrained vectors, not counts, and unknown words are rare.

**On slide (short)**
1. GloVe 6B-300d is our main embedding: the family the SNLI authors used.
2. Frozen BERT-base-uncased is the contextual alternative; never trained on NLI, so no leakage into CV.
3. We matched the tokenizer to each model (treebank for GloVe, WordPiece for BERT), so a GloVe–BERT gap belongs to the pair.
4. Coverage was a non-issue: the misses are rare words, only 0.24% (GloVe) and 2.8% (BERT, split) of tokens.
5. Compared with Project 1 (BPE-8k bag of words): words are now pretrained vectors, not counts, and unknown words are rare.

**Table** (2 rows × 5 columns; plain Keynote table, no shading)

*Caption:* Both vocabularies cover more than 99.7% of SNLI tokens. Only the rare tail is missing (GloVe) or split into pieces (BERT).

| | Tokenizer | Vocabulary | Words not in vocabulary | Sentences truncated |
|---|---|---|---|---|
| GloVe 6B-300d | treebank, lowercased | 400k words | 0.24% of 6.12M tokens (23% of 36,702 types) | none |
| BERT-base-uncased | WordPiece | 30,522 subwords | 0 [UNK] / 6.36M tokens | 26 of 628,646 (0.004%) at 64 tokens |

**Table 2: one SNLI pair, tokenized for each model** (train, label neutral; * = no GloVe vector)

| | GloVe (treebank) | BERT (WordPiece) |
|---|---|---|
| Premise | a young man with long blond-hair* skateboards down an inclined road . | a young man with long blond - hair skate ##boards down an inclined road . |
| Hypothesis | a young man is going to pop an ollie down the road . | a young man is going to pop an ollie down the road . |

*Caption:* GloVe keeps whole words but misses a rare compound (it gets a trained random vector); BERT splits rare words into known pieces. Common words are identical in both.

**Sources:** `nli/embeddings.py` (glove.6B.300d); `results/snli-glove-bowman-*` configs; Emil's coverage stats (`emil/`). GloVe stats: all 628,646 unique SNLI train sentences, treebank + lowercase (script: `results/slides/build/glove_stats.py`, not committed). Of the 8,452 missing types, 6,627 (78%) occur once.

**Pipeline detail (notes, not bullets):** missing words are *not* mapped to `<unk>`. The vocabulary is built from each training fold, and train words without a GloVe vector get a random N(0, 0.1) vector that is trained along with the rest (`nli/embeddings.py:39`). Only held-out words never seen in training become `<unk>`: 0.25% of dev tokens.

**Correction:** the 0.968 type coverage in slides_todo came from a 5k sample. On full train it is 0.770, because the rare tail grows with the data.

**Tentative (Emil, not yet backed by our data):** "casing carries almost no signal" (uncased model is fine) and "SNLI captions are well within BERT's training domain". Keep both out of the bullets and in the notes.

**Likely exam follow-ups**
- *"Did you use the same GloVe as Bowman et al.?"* No. They used 840B (Common Crawl, cased, 2.2M vocabulary); we use 6B (uncased, 400k vocabulary). Same method, smaller release, chosen for efficiency (≈1 min load, 0.5 GB in memory). Missing words are only 0.24% of running text even with the smaller vocabulary.
- *"How does treebank tokenization fit GloVe's definition of a token?"* Treebank splits punctuation and clitics ("n't", "'s"), close to how GloVe's training corpus was tokenized; we lowercase to match the uncased 6B vocabulary (`nli/tokenizers/treebank.py:22`).

---

## Slide 2: What do the embeddings capture, and what do they miss for NLI?

**Bullets** (full text, for us)
1. **Both models capture relatedness, not similarity.** Unrelated pairs come out lowest in both, but contradiction pairs score as high as synonyms or higher. man/woman beats man/guy in both GloVe (.70 vs .55) and BERT (.86 vs .83).
2. **This comes from how both are trained, not from one model.** Antonyms appear in the same contexts, so distributional training puts them close together, yet they are exactly the words that signal contradiction.
3. **Negation barely moves frozen BERT** (mean-pooled sentences): "A man is not sleeping" is about as close to "A man is sleeping" (.86) as an unrelated sentence is (.85). The GloVe models don't mean-pool, so this probe doesn't apply to them directly.
4. **BERT cosines sit in a narrow, high band (.42–.92) and GloVe's are spread out (.23–.70)** (anisotropy; Ethayarajh, 2019). Compare rankings within a column, never raw values across columns.
5. **Evidence from the task:** like for like (frozen, mean-pooled), BERT beats GloVe (.721 vs .706, +1.5 pts). The gain comes from relating the two sentences, not from artifacts: hypothesis-only BERT reaches only .618 (−10.3 pts). Matching features (|u−v|, u*v) lift BERT to .735. Fine-tuned GloVe with max pool is still best (.753): how we pool and whether we fine-tune matter more than which embedding we use. [ ] GloVe hypothesis-only control pending.

**On slide (short)**
1. Both models capture relatedness, not similarity: man/woman scores above man/guy in both.
2. Antonyms share contexts, so distributional training puts them close, yet they signal contradiction.
3. Negation barely moves frozen BERT: "not sleeping" vs "sleeping" (.86) is as close as an unrelated sentence (.85).
4. BERT cosines sit in a narrow high band (anisotropy): compare rankings, not raw values.
5. Like for like, BERT beats GloVe by 1.5 pts and uses the premise (+10 pts over hypothesis-only), but fine-tuned GloVe with max pool is still best.

**Table 1: word-pair cosine** (3 groups × 2 models; keep the group label in its own column)

*Caption:* In both models, contradiction pairs score as high as synonyms: cosine measures relatedness, not similarity.

| | Pair | GloVe 6B | BERT (word alone) |
|---|---|---|---|
| Synonyms | man / guy | .545 | .832 |
| | dog / puppy | .594 | .770 |
| | running / jogging | .243 | .845 |
| Contradiction cues | sitting / standing | .641 | .915 |
| | asleep / awake | .613 | .805 |
| | man / woman | **.700** | .862 |
| Unrelated | dog / car | .291 | .624 |
| | guitar / beach | .230 | .418 |

**Table 2: same Bowman model, held-out CV accuracy (2 seeds × 5 folds); combiner [u, v] unless noted**

*Caption:* Like for like, BERT beats GloVe by 1.5 pts and gains 10 pts from seeing the premise. Matching features (|u−v|, u*v) add 1.4 more, but fine-tuned GloVe with max pool is still best.

| Embedding | Premise + hypothesis | Hypothesis only |
|---|---|---|
| GloVe, mean pool, frozen | .706 | [ ] E1.5 (needs code) |
| BERT, mean pool, frozen | .721 | .618 |
| BERT, mean pool, frozen, + matching features | .735 | n/a |
| GloVe, max pool, trainable (best) | **.753** | [ ] E1.5 |

Like for like: rows 1–2 are both frozen and mean-pooled. The tokenizer still differs (treebank vs WordPiece), so the +1.5 belongs to tokenizer + embedding together (slide 1). BERT's pooling happens before the 768→100 projection, GloVe's after.

**Sources:** GloVe cosines from `results/slides/build/glove_stats.py` (glove.6B.300d, lowercased). BERT cosines and sentence probe from Emil (not re-run). Accuracy recomputed from folds.csv: `results/snli-glove-bowman-mean-freeze_embeddings=True-0074b3`, `results/snli-bert-bowman-combiner=concat-7d786240` (seeds .719 / .723), `results/snli-bert-bowman-combiner=hypothesis_only-2af11abe` (.625 / .612), `results/snli-bert-bowman-combiner=matching-0c24c9b3` (.737 / .733), `results/snli-glove-bowman-max-freeze_embeddings=False-b6d749c2`.

**Notes**
- running/jogging is low in GloVe (.24) because *running* is ambiguous (running a company, running for office). Static vectors average over every sense. Good example of a limitation BERT's context can fix.
- The BERT word cosines embed each word **alone**, but in our model BERT sees full sentences. Say so if asked.
- Emil's claim that the classifier recovers these signals via |u−v| and u*v now has support for BERT: +1.4 pts (86% better at ±0.5 pt). [ ] GloVe matching run (E1.3).
- Hypothesis-only BERT (.618) is *below* hypothesis-only BoW (.637, Project 1): frozen mean-pooled BERT vectors expose single cue words (*not*, *sleeping*) less directly than word counts do. Its fold SD is high (.023), so treat the exact value loosely.
- Artifacts check (BERT): full − hypothesis-only = +10.3 pts (concat), 100% outside the ROPE.

**Likely exam follow-ups**
- *"Is BERT better than GloVe for your task?"* Under equal conditions (frozen, mean pool), yes: +1.5 pts. But we only use BERT frozen, as fixed sentence vectors. Fine-tuned GloVe with max pool is still better (.753). Fine-tuning BERT would very likely change that, but it's outside our scope. ROPE test (±0.5 pt, seeds averaged per fold): 81% BERT better, 13% equivalent, 6% GloVe better. Likely better, not certain. Best BERT (.735, matching) vs best GloVe (.753, max pool): 91% GloVe max pool better.
- *"Why is anisotropy a problem?"* If all vectors point roughly the same way, cosine differences are small, so a classifier has to pick up small offsets. And as we use BERT, its vectors are frozen, so they can't adapt to the task.

---

## Slide 3: Embed-and-pool. Which choices matter, and did we beat Project 1?

**Bullets** (full text, for us)
1. **Model (Bowman et al., 2015):** each sentence is encoded separately by the *same* encoder, the two vectors are concatenated, and an MLP classifies the pair. The encoder is the only part we swap: sum, mean or max pool here, and an LSTM/GRU on the next slide.
2. **Protocol:** 5-fold CV × 2 model seeds, with the same folds (data_seed 0) for every run. We report held-out accuracy averaged over all 10 runs. Training defaults follow Bowman (AdaDelta, dropout 0.1, early stopping on 10% of the training fold).
3. **Pooling matters most:** max .753 > mean .733 > sum .722 (trainable). Max keeps the strongest feature from any word, while sum and mean blur a single contradicting word into the average.
4. **Fine-tuning the embeddings helps for every pool:** +1.4 (max), +2.7 (mean), +2.3 (sum). The frozen mean run is also the least stable (fold SD .021, against ≤ .012 for the others).
5. **Max pool beats Project 1's best (bow-mlp-512) by +2.6 pts** (pretrained embeddings + sentence encoder) and the hypothesis-only baseline by +11.6.

**On slide (short)**
1. Each sentence goes through the same encoder, then concat + MLP (Bowman et al., 2015). We only swap the encoder.
2. 5-fold CV × 2 seeds, same folds for every run; we report held-out accuracy.
3. Pooling matters most: max .753 > mean .733 > sum .722.
4. Fine-tuning GloVe helps every pool (+1.4 to +2.7 pts).
5. Max pool beats Project 1's best by 2.6 pts (new embeddings + sentence encoder).

**Figure: architecture** (built from Keynote shapes, left to right, as in the reference image)
- Two input rows: "Premise" and "Hypothesis".
- Each goes through **[GloVe 300d] → [tanh proj. 100d] → [Encoder]**. Draw a single *Encoder* box shared by both rows, or two boxes with a "shared weights" bracket, labelled "sum / mean / max  |  LSTM / GRU".
- Two 100d vectors **u**, **v** → **[concat 200d]** → **[3 × tanh 200]** → **[softmax: E / N / C]**.
- Mark the encoder box in the accent colour: "what we vary".

*Caption:* Premise and hypothesis go through the same encoder. The encoder is the only part we change across slides 3–4.

**Table: held-out CV accuracy (mean of 2 seeds × 5 folds)**

*Caption:* Max pool is the best pooling and beats Project 1's best by 2.6 pts. Fine-tuning GloVe helps every pool.

| Model | Trainable emb. | Frozen emb. |
|---|---|---|
| Hypothesis-only BoW log-lin (P1) | .637 | n/a |
| BoW MLP-512 (P1 best) | .727 | n/a |
| GloVe + sum pool | .722 | .699 |
| GloVe + mean pool | .733 | .706 |
| GloVe + max pool | **.753** | .739 |

**Sources:** `results/snli-glove-bowman-{sum,mean,max}-freeze_embeddings=*`, `results/snli-bow-mlp-hidden_dim=512-n_blocks=0-38b6d7fe`, `results/snli-hyp-only-bow-loglin-8059a0af`. Recomputed from folds.csv on 2026-11-20.

**Notes**
- If asked how much of the +2.6 comes from the embedding versus the architecture: the random-init run (E1.4) separates the two. [ ]
- Hypothesis-only seed 1 is a copy of seed 0 (lbfgs is deterministic), so its "2 seeds" are really 1.
- Artifacts, mentioned briefly: hypothesis-only reaches .637 without seeing the premise (Project 1). Every gain above that line should come from relating premise and hypothesis, which the hypothesis-only Bowman runs (E1.5) will test.
- Sum pool is the paper's baseline. Bowman report 75.3 test for 100d sum of words with GloVe 840B. Not comparable: theirs is on the test set, ours is CV on train folds.
- Not tested: embedding size, optimiser settings. Both are dropped from the experiment queue. If asked, we followed the paper's defaults and spent the budget on the encoder.

**Likely exam follow-ups**
- *"Why does max beat mean?"* NLI often depends on one word (*not*, *woman*, *sleeping*). Max over each dimension keeps that word's strongest feature; the mean shrinks it by 1/length.
- *"Is +2.6 meaningful?"* It's about 5× the fold SD of max pool (.005) and the seeds agree (.753 / .753). Slide 5 covers this properly.

---

## Slide 4: Recurrent encoder. How far does an LSTM/GRU go?

**Bullets** (full text, for us)
1. **Swapping the pool for a 100d LSTM (last hidden state) ties max pool:** .754 vs .753. 72% within the ±0.5-pt ROPE, so word order isn't adding anything we can measure yet. The GRU (E1.1) and pooling over all RNN states (E1.2) are the next attempts. [ ]
2. **Freezing GloVe hardly matters for the LSTM:** .754 trainable vs .751 frozen, and the winner flips with the seed (seed 0: frozen .755 > trainable .751; seed 1 the other way). For the pools it matters more (+1.4 to +2.7 pts).
3. **Mild overfitting, controlled by early stopping:** at the epoch we keep, train runs ahead of validation by 2.2 pts (LSTM) and 1.7 (max pool). Bowman's LSTM had a 7-pt train–test gap (.848 / .776). Runs stop at about epoch 16 of 30.
4. **Data size:** [ ] learning curve (E3). Does the LSTM overtake max pool as the training set grows?
5. **How far:** our best is .754 (CV), against Bowman's .776 (test, GloVe 840B, full training set). Our models train on about three quarters of SNLI (4/5 folds × 90%). [ ] the learning curve tells us whether that explains the gap.

**On slide (short)**
1. An LSTM ties max pool (.754 vs .753): word order adds nothing measurable yet.
2. Freezing GloVe hardly matters for the LSTM; the winner flips with the seed.
3. Overfitting is mild: early stopping keeps train within ~2 pts of validation.
4. Data size: TBD. Does the LSTM overtake max pool with more data?
5. Our best is .754 (CV) vs Bowman's .776 (test); we train on about three quarters of SNLI.

**Figure 1: training curve** (two panels side by side, or one panel with 4 lines)
- x: epoch (1–20); y: accuracy.
- Lines: LSTM train and val (solid and dashed, colour A); max pool train and val (colour B).
- Average over the 10 runs (2 seeds × 5 folds); stop the line at the epoch where fewer than 5 runs are still training. Mark the mean best epoch (LSTM 13.2, max 13.7) with a thin vertical tick.
- *Caption:* At the epoch we keep, train and validation accuracy are within about 2 pts, so early stopping keeps overfitting mild. (Train accuracy is measured with dropout on, so it understates the true value.)
- Data: `results/snli-glove-bowman-{lstm,max}-freeze_embeddings=False-*/history.csv`.

**Figure 2: learning curve** [ ] E3
- x: training fraction {10, 25, 50, 100}% (log scale); y: held-out CV accuracy.
- Lines: max pool and best RNN, GloVe (solid) and random init (dashed). Error bars: SD over the 10 runs.
- Dashed horizontal reference lines: bow-mlp-512 (.727) and hypothesis-only (.637).
- *Caption:* [ ] write once E3 lands. It should answer: does the LSTM overtake max pool as data grows, and does GloVe matter more with less data?

**Sources:** `results/snli-glove-bowman-lstm-freeze_embeddings=*`, `...-max-freeze_embeddings=False-*`. Gaps at the best epoch, from history.csv, averaged over 10 runs (recomputed 2026-11-20). Bowman numbers: `papers/00_bowman2015_snli.pdf`, Table 6.

**Notes**
- The gaps in slides_todo (.041 / .032) are at the *last* epoch. The runner restores the best checkpoint (`nli/models/bowman.py:251`), so the best-epoch gap is the relevant one.
- The val accuracy in history.csv (.756) is the internal 10% early-stopping split, not the held-out fold.
- Frozen LSTM gap is only .006: freezing acts as a regulariser, yet doesn't cost accuracy.
- If the GRU lands: GRU vs LSTM row in the bullet 1 sentence, no new table.

**Likely exam follow-ups**
- *"Why doesn't the LSTM beat max pool?"* SNLI hypotheses are short (mean ≈ 10 tokens) and many labels hinge on one word, which max pool catches. And the last hidden state has to compress the whole sentence into 100d.
- *"Did you struggle with overfitting?"* Not much. Early stopping on a validation split keeps the gap at about 2 pts. If anything we may stop too early (Bowman's models fit train much more closely).

---

## Slide 5: Do we have enough evidence for our choices?

**Bullets** (full text, for us)
1. **Structured search:** one anchor (Bowman defaults, GloVe, trainable), then change one thing at a time: encoder → freezing → embedding → combiner. We skipped embedding size and dropout sweeps so the budget could go to folds × seeds, as the tutorial recommends.
2. **Paired runs:** every config uses the same 5 folds and the same 2 model seeds, so each comparison is like for like, fold by fold.
3. **Seeds matter more than folds for the neural models:** for the LSTM the seed SD is 0.95 pts against a fold SD of 0.53, and the two seeds differ by up to 3.1 pts on one fold (sum pool). With a single seed, the frozen-vs-trainable LSTM result would have come out either way.
4. **Decisions use the correlated Bayesian t-test (Benavoli et al., 2017), seeds averaged per fold, with a ROPE of ±0.5 pt as in tutorial 1:** max > mean (99%), LSTM > sum (99%), trainable > frozen for sum (95%) and max (97%). LSTM ≈ max pool (72% within the ROPE). LSTM trainable vs frozen is undecided (36% trainable better, 55% within the ROPE).
5. **Final check:** refit the chosen configs on the full training set and evaluate **once** on dev and test (single models; ensembles marked). [ ]

**On slide (short)**
1. Structured search: one anchor, then change one thing at a time.
2. Paired runs: the same folds and seeds for every config.
3. Seeds matter more than folds: with one seed, a conclusion could have flipped.
4. Bayesian correlated t-test, ROPE ±0.5 pt: pooling and fine-tuning are clear; LSTM vs max pool is a tie.
5. Final check: evaluate the chosen configs once on dev and test (TBD).

**Figure: one horizontal stacked bar per decision** (P(left better) | P(within ROPE) | P(right better); 3 muted colours, % printed in each segment)
- Rows (P left / ROPE / P right): max vs mean .99 / .01 / .00; LSTM vs sum .99 / .00 / .00; max trainable vs frozen .97 / .03 / .00; sum trainable vs frozen .95 / .03 / .01; BERT vs GloVe (both frozen, mean) .81 / .13 / .06; BERT matching vs concat .86 / .11 / .03; BERT full vs hypothesis-only 1.00 / .00 / .00; LSTM trainable vs frozen .36 / .55 / .09; LSTM vs max .17 / .72 / .11.
- Sort by P(within ROPE), so the clear decisions sit at the top and the ties at the bottom.
- [ ] Add later: GRU vs LSTM, GloVe full vs hypothesis-only.
- Axis label: "posterior probability, ROPE ±0.5 pt accuracy".

*Caption:* Pooling and fine-tuning are clear decisions. LSTM vs max pool is a tie, and whether to freeze the LSTM's embeddings is still undecided.

**Table: final evaluation** [ ] (once, at the end)

*Caption:* [ ] write after the final evaluation: does the CV ranking hold on dev and test, and how far are we from Bowman's .776?

| System | CV (2 × 5) | Dev | Test |
|---|---|---|---|
| Hypothesis-only BoW (P1) | .637 | [ ] | [ ] |
| BoW MLP-512 (P1 best) | .727 | [ ] | [ ] |
| GloVe + max pool | .753 | [ ] | [ ] |
| Best RNN | [ ] | [ ] | [ ] |
| BERT + matching features | .735 | [ ] | [ ] |
| Bowman et al. LSTM (lit.) | n/a | n/a | .776 |

**Sources:** variance from `results/figures/point5/variance_seed_vs_fold.csv` (Dario). ROPE rows from `results/slides/build/rope_seedavg.py` (seed-averaged, `baycomp.two_on_single(..., runs=1)`), recomputed 2026-11-20.

**Notes**
- **Why seeds are averaged:** the tutorial's `runs` are repeated CV on *different* fold splits (`tutorial.py:1308`). We vary only the model seed and keep the folds fixed, so our two seeds aren't independent CV repeats. We average them per fold and test with `runs=1`. Compared with Dario's `runs=2` numbers, no decision changes; close calls move slightly toward the ROPE (LSTM trainable vs frozen: 45% → 55%). [ ] Tell Dario so `compare.py` and his CSVs match.
- Robustness check at ±1 pt (Dario): the clear decisions don't change; LSTM trainable vs frozen moves to 85% within the ROPE, i.e. "equivalent". Close calls depend on the ROPE size, so state it on the slide.
- Hypothesis-only seed 1 is a copy of seed 0, so that row has no seed variance.
- Folds share 75% of their training data, so fold scores aren't independent. That's why we use the correlated t-test rather than a plain one.

**Likely exam follow-ups**
- *"Why 2 seeds and not more?"* Budget: ~1–2 GPU-hours per config. The variance table shows seeds matter, so for the final decision (best RNN vs max pool) we'd add seeds before folds.
- *"Why a ROPE?"* A difference can be real but too small to matter; the ROPE lets us conclude "equivalent", which a p-value can't.

---

## Slide 6: Error analysis. Which inputs separate our systems?

**Bullets** (full text, for us)
1. **Analysis features** (SNLI dev, 900 items stratified by label, every fold × seed model): hypothesis and premise length, word overlap (McCoy et al., 2019), negation in the hypothesis, and **cue words**: the top-15 PMI hypothesis words per label on train (Gururangan et al., 2018). Each item is flagged by whether its cue points to the gold label or to a wrong one ("misleading").
2. **Cue words are the clearest test for artifacts.** When a cue misleads, the hypothesis-only and bag-of-words models fall to ≈.32–.37 accuracy, and the LSTM holds ≈.60. The BDA agrees: LSTM − BoW log-lin on misleading cues is +0.17 (P > 0 = 1.00). Some of the gain comes from reading the premise, not from the cues.
3. **Negation (slide 2's limitation) hurts the sentence encoders relative to BoW:** LSTM − BoW is −0.14 (P > 0 = .007). Negated hypotheses are mostly contradictions (4.2% vs 0.7–0.8%), so the BoW models score ≈.89 by using "not" as a cue, while the LSTM gets ≈.81.
4. **LSTM vs sum pool:** no feature is clearly different (all 95% intervals include 0). Order sensitivity doesn't show up in these features. The only hint: the LSTM is slightly worse on long hypotheses (P > 0 = .03).
5. **Next priorities:** (a) an antonym/swap feature (sitting/standing, man/woman), the other limitation from slide 2; (b) max pool, GRU and BERT in the BDA (BERT: does context fix negation?) [ ] rerun on Snellius; (c) models that compare words across the two sentences (attention), since our encoders compress each sentence before the comparison.

**On slide (short)**
1. Features: length, word overlap, negation, and hypothesis cue words (Gururangan et al., 2018).
2. When a cue word misleads, BoW models collapse (≈.35) and the LSTM holds ≈.60: it reads the premise.
3. Negation favours BoW: "not" is a reliable contradiction cue that BoW exploits. Does BERT's context help? TBD.
4. LSTM vs sum pool: no feature separates them.
5. Next: an antonym feature, more systems in the analysis, and cross-sentence attention.

**Table: accuracy on two subsets** (draft: values read off Dario's `accuracy_by_bin.png`, ≈; replaces the figure until exact numbers exist)

| Subset | Hyp-only | BoW log-lin | Sum | LSTM | BERT |
|---|---|---|---|---|---|
| Negated hypothesis | ≈.89 | ≈.89 | ≈.83 | ≈.81 | [ ] |
| Cue word points to a wrong label | ≈.32 | ≈.37 | ≈.54 | ≈.60 | [ ] |

*Caption:* When a hypothesis cue word points to the wrong label, the hypothesis-only and BoW models collapse while the LSTM holds ≈.60. On negated hypotheses, BoW wins.

**(Later) Figure 1: accuracy by feature subset** (redraw 2 of Dario's 6 panels: *negation* and *cue_misleading*)
- x: feature 0 / 1; y: accuracy (.3–1.0); one dot + error bar (SD over models) per system.
- Systems: hyp-only, BoW log-lin, BoW MLP, sum, LSTM. [ ] Add max pool.
- [ ] Exact values: regenerate `accuracy_by_bin` as a CSV on Snellius (the numbers in the bullets are read off the plot, ≈).
- *Caption:* When a hypothesis cue word points to the wrong label, the hypothesis-only and BoW models collapse (≈.32–.37) while the LSTM holds ≈.60. On negated hypotheses, BoW wins.

**Figure 2: BDA weight differences** (forest plot: feature on y; posterior mean ± 95% interval on x; a vertical line at 0)
- Two comparisons: LSTM − BoW log-lin, and LSTM − sum. Features: overlap, negation, cue_gold, cue_misleading, hypothesis_len.
- Data: `results/figures/bda-v1.1/weight_differences_all.csv`.
- *Caption:* Compared with BoW, the LSTM relies less on cue words and does better when the hypothesis overlaps with the premise. LSTM and sum pool differ on no feature.

**Sources:** `results/figures/bda-v1.1/` (Dario, commit 627d8de): `weight_differences_all.csv`, `feature_rates_by_label.csv`, `accuracy_by_bin.png`, `diagnostics.json` (max R-hat 1.005, 0 divergences).

**Notes**
- Follows the tutorial's method: an analysis split (SNLI dev, while test stays untouched), input features → correctness, and the tutorial's numpyro BDA model (`nli/analysis.py:302`). One deliberate change: Dario conditions on the **gold** label instead of the predicted label, because the predicted label is a model output, not an input property.
- **Caveat (if asked):** as in the tutorial (`tutorial.py:1551`), every fold × seed model contributes its own predictions, so each dev item appears 10 times per system. This keeps the run-to-run variation in the analysis, but the model has no per-item term, so repeated rows count as extra evidence and the intervals are somewhat narrow. Read the weights as trends, not exact probabilities.
- cue_gold, LSTM − BoW: −0.41 (P > 0 = 0). The LSTM gains *less* from helpful cues, which fits the same story as bullet 2.
- Overlap: LSTM − BoW +0.13 (P > 0 = .99). The LSTM does relatively better when the hypothesis reuses premise words.
- "LSTM" and "sum" here are the GloVe trainable Bowman runs.

**Likely exam follow-ups**
- *"Is the LSTM better because it uses the premise, or because of artifacts?"* Misleading cues: the hypothesis-only model has to fail on them (.32), and the LSTM is at .60. The hypothesis-only LSTM (E1.5) will make this direct.
- *"Why is negation worse for the LSTM?"* The BoW models treat "not" as a strong contradiction cue, which is right most of the time on SNLI. The LSTM has to carry it through the sentence state, and pretrained vectors put "not" close to its context (slide 2 showed this for BERT; we haven't probed it for GloVe).
