# ESc1 take-home

## Setup

```bash
pip install -r requirements.txt
```

## EDA

Run interactively:

```bash
PYTHONPATH=src jupyter notebook notebooks/01_eda.ipynb
```

Or execute end-to-end:

```bash
PYTHONPATH=src jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --inplace
```

Run helper tests:

```bash
PYTHONPATH=src pytest -q
```

Reusable statistics live in `src/takehome/eda.py`; the notebook is only the research narrative and plotting layer.

The committed notebook includes evaluated outputs and figures from the full LFS dataset.
