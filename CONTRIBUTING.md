# Contributing to RoCast

Thanks for your interest in RoCast! This project forecasts solar and wind generation for the Romanian SEN (ENTSO-E bidding zone `10YRO-TEL------P`) with probabilistic (P10–P90) outputs.

## Getting started

```bash
python -m venv .venv && .venv/Scripts/activate        # Windows
# python -m venv .venv && source .venv/bin/activate    # Linux/macOS
pip install -e ".[dev,nowcast]"
pre-commit install                                     # optional but recommended
```

## Workflow

1. Fork and create a feature branch.
2. Make your change. Keep it small and focused.
3. Run the checks locally before pushing:

```bash
ruff check rocast tests scripts
ruff format --check rocast tests scripts
pytest -v
python -m rocast.cli backtest --mode synthetic --epochs 3 --output-dir outputs_check
```

4. Open a pull request. CI (`.github/workflows/ci.yml`) runs the same checks on Python 3.11 and 3.13.

## Code style

- `ruff` with line length 120, rules `E, F, I, W, UP, B` (see `[tool.ruff]` in `pyproject.toml`). Pre-commit runs `ruff --fix` and `ruff format`.
- Type hints on public functions; docstrings on public classes/modules.
- No new heavy dependencies without discussion — the core must stay installable with `pip install -e .`.

## Tests

- Tests live in `tests/` and run with `pytest`. The end-to-end test uses `--mode synthetic`, so no API keys are needed.
- Add a test for any new behavior you introduce.

## Live data and secrets

- Never commit API keys. `.env` is gitignored; copy `.env.example` to `.env` locally.
- Live mode (`--mode live`) needs ENTSO-E and optionally NGC/EUMETSAT keys and is not covered by CI.

## Reporting issues

Please include: RoCast version (`rocast --version`), Python version, OS, the exact command, and the full error traceback.
