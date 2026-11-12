# RoCast 🇷🇴⚡

![python](https://img.shields.io/badge/python-3.11%2B-blue) ![license](https://img.shields.io/badge/license-MIT-green) ![tests](https://img.shields.io/badge/tests-pytest-brightgreen) ![zone](https://img.shields.io/badge/ENTSO--E-10YRO--TEL------P-orange)

Multi-scale **probabilistic Solar (PV) and Wind** forecasting for the Romanian National Power System (SEN), built for day-ahead / intraday balancing venues such as **Predico**, **EnergyBench** and **OPCOM**. 15-minute MTU, 6 h nowcast, 48 h day-ahead, quantiles P10–P90. No generative LLMs in the loop.

![RoCast hero](assets/rocast_linkedin_hero.png)

## Romanian prosumer split

| Source | ENTSO-E doc | What it captures |
|---|---|---|
| Utility solar, onshore wind | **A75** | metered generation (`solar_mw`, `wind_onshore_mw`) |
| Total load | **A65** | includes ~4 GW behind-the-meter prosumer PV self-consumption |

RoCast models both and derives `net_load_ex_renewables_mw = total_load_mw − solar_mw − wind_onshore_mw`.

## Three branches

1. **pySTEPS + pvlib** — Heliosat cloud index from EUMETSAT, Lucas-Kanade optical flow + semi-Lagrangian advection (24 steps = 6 h), multiplied by Ineichen clear-sky GHI at five hubs (Dobrogea, Focșani/Moldova, Wallachia, Oltenia, Transylvania). Automatic NumPy fallback if pySTEPS is unavailable.
2. **NVIDIA Earth-2** — `earth2studio` FourCastNet 3 ensembles, NIM `/v1/infer` tar client, cropped to Romania; ensemble mean and spread maps. A deterministic synthetic ensemble powers offline runs.
3. **`PVNetRomaniaFusion`** — PVNet-style 3D satellite encoder + Earth-2 spatial encoder + GRU telemetry + exogenous MLP, strictly monotone quantiles, capacity and night clamps, pinball loss. See [PLAN.md](PLAN.md) for tensor shapes and cutoff math.

## Install

```bash
uv venv && uv pip install -e ".[dev,nowcast]"     # or: python -m venv .venv && pip install -e ".[dev,nowcast]"
# optional: pip install earth2studio satpy chronos-forecasting opencv-python-headless
cp .env.example .env                               # ENTSO-E / NGC keys for live mode
```

## Usage

```bash
python -m rocast.cli backtest --mode synthetic --epochs 3 --output-dir outputs
python scripts/generate_linkedin_hero.py
pytest -v
```

Outputs: `outputs/predico_submission.csv` (`datetime,q10,q25,q50,q75,q90`, ISO-8601 UTC; target chosen with `--target`), per-target `predico_submission_<target>.csv`, and `outputs/backtest_metrics.json`.

`--mode live` pulls ENTSO-E (A75/A65) and Open-Meteo for the last `--days` days; pass `--seviri-dir` (SEVIRI native files) and `--earth2 earth2studio` to activate the satellite and Earth-2 branches, otherwise they are disabled with a warning.

## Benchmark (synthetic data, 3 epochs, 3 walk-forward test days)

| Target | Pinball (MW) | CRPS (MW) | MAE P50 (MW) | P10–P90 coverage | Skill vs NWP |
|---|---|---|---|---|---|
| Solar | 55.2 | 110.4 | 158.8 | 96.4 % | −75.9 % |
| Wind onshore | 118.5 | 237.1 | 351.6 | 77.1 % | −9.7 % |

**Honest note:** these figures are on a *synthetic* world with a deliberately strong deterministic NWP baseline, a tiny training set (60 overlapping windows) and only 3 epochs, so the model is under-trained and does **not** yet beat the baseline (more epochs/data are expected to help, but this is untested here). The numbers validate the pipeline mechanics, not real-world Predico performance. Run `--mode live` with real data for genuine evaluation.
