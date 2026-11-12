# Changelog

All notable changes to RoCast are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-12

### Added

- Initial public release: multi-scale probabilistic Solar/Wind forecasting engine for the Romanian SEN (`10YRO-TEL------P`).
- Three-branch architecture: pySTEPS + pvlib clear-sky advection nowcasting, NVIDIA Earth-2 (FourCastNet 3) ensemble branch, and the `PVNetRomaniaFusion` spatio-temporal fusion model with strictly monotone P10–P90 quantiles.
- Walk-forward backtest CLI (`rocast backtest`) with synthetic offline mode and live ENTSO-E/Open-Meteo mode, Predico-compatible submission export (`datetime,q10,q25,q50,q75,q90`, ISO-8601 UTC).
- Predico evaluator: pinball loss, CRPS, MAE/RMSE of the median, P10–P90 coverage, and skill score vs the NWP baseline.
- 15-minute MTU resolution, 6 h nowcast horizon, 48 h day-ahead horizon, five Romanian hubs (Dobrogea, Moldova/Focșani, Wallachia, Oltenia, Transylvania).
- Test suite (pytest), CI (GitHub Actions, Python 3.11/3.13), pre-commit hooks (ruff), MIT license, packaging (`python -m build` produces sdist + wheel).
- `scripts/generate_linkedin_hero.py` renders the hero figure in `assets/`.
- Cosine-annealing LR schedule and defaults of 45 epochs / 21 days / train stride 4 / 5 test windows for the fusion model.

### Changed

- Fusion head is now a residual on the NWP baseline: quantiles start exactly at the NWP fan (zero-init output layer) and the model only learns monotone spread increments plus a median correction, so uncertain predictions can never collapse to zero.
- The learned median correction is gated by the nowcast horizon (full 24 steps for solar, 8 for wind): beyond the satellite advection horizon the median falls back to NWP, which removed the train/test overfit gap (test skill went from negative to strongly positive for solar).
- Synthetic Earth-2 ensemble now uses an NWP-like wind (2 h lag + smoothing) instead of the true future wind, removing a synthetic-mode leakage in the ensemble branch.
- Per-timestamp caching of satellite / Earth-2 / nowcast fields in the window builder: window builds repeat every epoch, so each underlying field is now computed once (~5x faster epochs after the first).
- CLI: `--version`, `--quiet`, `--n-test-windows`, `--train-stride`, epilog examples, and clean exit code 1 with a logged error on bad input.

### Notes

- Live mode (real ENTSO-E/SEVIRI/Earth-2 inputs) requires API keys and is not exercised by the offline test suite.
- Benchmarks on the synthetic world are reported in the README; they validate pipeline mechanics, not real-market skill.
