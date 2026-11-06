# Contributing

Thanks for taking a look at this project.

## Development setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

## Before opening a pull request

```bash
pytest
python -m compileall -q src
```

If you change the attack logic, please include a regression test or update
`tests/test_attack.py`. If you change reported results, regenerate the files in
`results/` and `visualizations/` with the command documented in the README so
the checked-in outputs stay consistent.

## Scope

Useful contributions include:

- additional attack variants (PGD, targeted FGSM),
- more surrogate/target algorithm combinations,
- additional metrics or statistical summaries,
- reproducibility improvements and documentation fixes.

Please keep changes focused and avoid committing model checkpoints. They are
downloaded via `python -m rai.download_models`.
