"""NLI experiment framework.

Functional core / imperative shell:

- Core (pure, no I/O): ``folds.make_folds``, ``Representation.fit/transform``,
  ``Model.fit/predict_proba``, ``metrics.score``, ``ensemble.combine``,
  ``compare.compare``.
- Shell (all I/O): ``data.load_dataset``, ``experiment.run_experiment``,
  ``store.*``, ``ensemble.ensemble_on_dev``.

See ``.scratch/project1/spec.md`` for the architecture and ``GLOSSARY.md`` for vocabulary.
"""
