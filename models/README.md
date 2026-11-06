Fitted Fold models (`<experiment id>/seed-S-fold-K.*`) and trained tokenizers (`tokenizers/`). Everything here except this file is gitignored; `run_experiment` regenerates it from a config.

Hyperparameters

Any key in a config block other than `kind` is passed to that class as a keyword argument, so all of these can go in a grid (`grids/*.json`, `scripts/run_grid.py`). Defaults in brackets.

| block | kind | hyperparameter |
|---|---|---|
| tokenizer | `wordlevel` | `min_frequency` [1], `lowercase` [true] |
| tokenizer | `bpe` | `vocab_size` [8000], `min_frequency` [2] |
| representation | `bow` | `pairing` [concat], `binary` [false], `normalize` [false] |
| representation | `ngram` | `pairing` [concat], `n` [2], `min_df` [1], `binary` [false], `max_features` [null] |
| representation | `lengths` | `pairing` [concat] |
| model | `loglinear` | `C` [1.0], `l1_ratio` [0.0], `max_iter` [1000] |
| model | `mlp` | `hidden_dim` [128], `dropout` [0.1], `n_blocks` [0], `residual` [true], `batch_size` [64], `learning_rate` [1e-3], `weight_decay` [1e-4], `num_epochs` [20], `validation_fraction` [0.1], `patience` [3] |

`pairing` is `concat` (premise features ++ hypothesis features), `joint` (one vector over both texts) or `hypothesis_only`.

Fixed design (not tuned)

- `loglinear`: sklearn `LogisticRegression`, multinomial; `l1_ratio` picks the penalty (0 = l2, 1 = l1). Pure l2 uses the `lbfgs` solver, anything with an l1 share uses `saga`.
- `mlp`, `n_blocks=0`: `Linear(in, hidden) -> ReLU -> Dropout -> Linear(hidden, 3)`.
- `mlp`, `n_blocks>0`: `Linear(in, hidden) -> ReLU -> n_blocks x [LayerNorm -> Linear -> ReLU -> Dropout] -> LayerNorm -> Linear(hidden, 3)`; each block adds its input back when `residual`.
- `mlp` training: Adam, cross-entropy, `validation_fraction` of the fold's training rows held out, early stopping on validation accuracy after `patience` epochs without improvement, best epoch's weights kept. `device` [cpu] is a config key too but not a hyperparameter.
- `majority`: predicts the training label distribution; no hyperparameters.
- Folds: stratified, `dataset.k` [5], `dataset.data_seed` [0], shared by every experiment on a dataset. `model_seeds` [[0]] seeds the model only.
