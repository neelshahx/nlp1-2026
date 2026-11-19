Fitted Fold models (`<experiment id>/seed-S-fold-K.*`) and trained tokenizers (`tokenizers/`). Everything here except this file is gitignored; `run_experiment` regenerates it from a config.

Hyperparameters

Any key in a config block other than `kind` is passed to that class as a keyword argument, so all of these can go in a grid (`grids/*.json`, `scripts/run_grid.py`). Defaults in brackets.

| block | kind | hyperparameter |
|---|---|---|
| tokenizer | `wordlevel` | `min_frequency` [1], `lowercase` [true] |
| tokenizer | `bpe` | `vocab_size` [8000], `min_frequency` [2] |
| tokenizer | `treebank` | `lowercase` [true], `min_frequency` [1] |
| representation | `bow` | `pairing` [concat], `binary` [false], `normalize` [false] |
| representation | `ngram` | `pairing` [concat], `n` [2], `min_df` [1], `binary` [false], `max_features` [null] |
| representation | `lengths` | `pairing` [concat] |
| representation | `glove` | `dim` [300], `min_frequency` [1] |
| representation | `bert` | `pairing` [matching], `model_name` [bert-base-uncased], `max_length` [64], `batch_size` [256], `device` [auto] |
| model | `loglinear` | `C` [1.0], `l1_ratio` [0.0], `max_iter` [1000] |
| model | `bowman` | `encoder` [lstm], `freeze_embeddings` [false], `proj_dim` [100], `hidden_dim` [100], `mlp_dim` [200], `n_layers_mlp` [3], `dropout` [0.1], `batch_size` [128], `learning_rate` [1.0], `weight_decay` [1e-6], `num_epochs` [30], `validation_fraction` [0.1], `patience` [3] |
| model | `mlp` | `hidden_dim` [128], `dropout` [0.1], `n_blocks` [0], `residual` [true], `batch_size` [64], `learning_rate` [1e-3], `weight_decay` [1e-4], `num_epochs` [20], `validation_fraction` [0.1], `patience` [3] |

`pairing` is `concat` (premise features ++ hypothesis features), `joint` (one vector over both texts) or `hypothesis_only`. `bert` takes `concat`, `matching` ([u, v, |u-v|, u*v]) or `hypothesis_only`, and ignores the experiment's tokenizer (it uses WordPiece).

Fixed design (not tuned)

- `loglinear`: sklearn `LogisticRegression`, multinomial; `l1_ratio` picks the penalty (0 = l2, 1 = l1). Pure l2 uses the `lbfgs` solver, anything with an l1 share uses `saga`.
- `mlp`, `n_blocks=0`: `Linear(in, hidden) -> ReLU -> Dropout -> Linear(hidden, 3)`.
- `mlp`, `n_blocks>0`: `Linear(in, hidden) -> ReLU -> n_blocks x [LayerNorm -> Linear -> ReLU -> Dropout] -> LayerNorm -> Linear(hidden, 3)`; each block adds its input back when `residual`.
- `mlp` training: Adam, cross-entropy, `validation_fraction` of the fold's training rows held out, early stopping on validation accuracy after `patience` epochs without improvement, best epoch's weights kept. `device` [cpu] is a config key too but not a hyperparameter.
- `bert`: frozen encoder, mean pooling over real tokens, nothing fitted; sentence vectors are memoised per process so folds share one encoding pass.
- `treebank`: NLTK Penn Treebank tokens, lowercased, as GloVe 6B was built; returns raw tokens (no `[UNK]`), so `glove` decides what is out of vocabulary.
- `glove`: vocabulary from the fold's training pairs only (`<pad>`=0, `<unk>`=1), rows initialised from `data/glove.6B/glove.6B.<dim>d.txt`, missing words N(0, 0.1). Outputs token-id sequences plus the matrix; only sequence models (`bowman`) consume it.
- `bowman` (Bowman et al. 2015, Fig. 3): embedding -> `Linear(300, proj_dim) -> tanh` -> shared sentence encoder -> `[u; v]` -> `n_layers_mlp` x `Linear -> tanh -> Dropout` -> `Linear(·, 3)`. `encoder`: `lstm` (last hidden state of a `hidden_dim` LSTM), `sum` (the paper's sum-of-words embed-and-pool baseline), `mean`, `max`. AdaDelta + L2, early stopping on a `validation_fraction` slice of the fold's training rows, as `mlp`. `device` [auto] = cuda, else mps, else cpu.
- `majority`: predicts the training label distribution; no hyperparameters.
- Folds: stratified, `dataset.k` [5], `dataset.data_seed` [0], shared by every experiment on a dataset. `model_seeds` [[0]] seeds the model only.
