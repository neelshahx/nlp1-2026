```
uv sync
uv run jupyter lab
uv run jupytext --to py:percent ${.ipynb name}

(OPTIONAL) uv run pytest
```

Within jupyter lab right-click on .py file to open as jupyter notebook
