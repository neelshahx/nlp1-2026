Common commands

```
uv sync
uv run jupyter lab
uv run jupytext --to py:percent ${.ipynb name}
uv run pytest
uv run ruff check --fix .
uv run ruff format .
uv run basedpyright nli/
```

Within jupyter lab right-click on .py file to open as jupyter notebook

Glossary

- Dataset (data.py): raw dataset
- Tokenizer (tokenizers/): turns text into tokens
- Representation (representations/): turns tokens into features, vector-valued or sequences
- Model (models/): trainable classifier that typically outputs posterior class probabilities
- Experiment (experiment.py): one configuration of (Dataset, Tokenizer, Representation, Model)
- Result (store.py): output of an experiment, such as scores and metrics
- Ensemble (ensemble.py): set of fitted models scored on dev set according to some Combiner, e.g. majority
- Comparison (compare.py): compare models within class between folds

Sample experiment config

```json
{
  "name": "toy-lengths-majority",
  "dataset": {
    "k": 3,
    "data_seed": 0,
    "rounds": null,
    "name": "toy"
  },
  "tokenizer": {
    "kind": "wordlevel",
    "min_frequency": 1
  },
  "representation": {
    "kind": "lengths",
    "pairing": "concat"
  },
  "model": {
    "kind": "majority"
  },
  "model_seeds": [
    0,
    1
  ],
  "metrics": [
    "accuracy",
    "macro_f1"
  ]
}
```