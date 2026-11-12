# RoCast — system plan

Multi-scale probabilistic Solar (PV) and Onshore-Wind forecasting for the Romanian SEN
(ENTSO-E bidding zone `10YRO-TEL------P`, `Europe/Bucharest`, 15-minute MTU). No generative LLM/VLM is used anywhere in the forecasting loop.

## 1. Architecture

```
            ┌────────────────────────── t0 cutoff gate ──────────────────────────┐
EUMETSAT ──► Heliosat cloud index n ─► pySTEPS LK flow ─► semi-Lagrangian ──► advected n (24,H,W)
 (SEVIRI)     (rho_min per pixel)       (dense NumPy LK fallback)                    │ 3x3 hub mean × Ineichen GHI
                                                                                     ▼
pvlib Ineichen clear-sky + solar geometry (5 hubs) ───────────────────► hub GHI nowcast 0-6 h
Earth-2 (FCN3 ensemble / NIM / synthetic) ─► (T,16,24,40) mean+spread maps + hub features
Open-Meteo NWP (9 vars + cubic wind proxy) ─► exogenous covariates
ENTSO-E A75 (solar, wind onshore), A65 (load) ─► 15-min telemetry (gaps ≤ 4 MTU interpolated)
                                    │
                                    ▼
                         PVNetRomaniaFusion  ─►  quantiles (B, 192, 5)  ─►  PredicoEvaluator / CSV
```

## 2. Tensor topology

| Branch | Tensor | Encoder | Output |
|---|---|---|---|
| Satellite + advection | `(B, 5, 8, 24, 40)` = 4 SEVIRI ch + cloud index; 4 observed + 4 advected frames | `SatelliteAdvection3DEncoder` (Conv3d→BN→SiLU→MaxPool→AdaptiveAvgPool) | `(B, d)` |
| Telemetry | `(B, 96, F_hist)` | 2-layer GRU (+ optional Chronos-Bolt prior hook on exo branch) | `(B, d)` |
| Earth-2 | `(B, 192, 16, 24, 40)` (8 vars × [mean, spread]) | `Earth2EnsembleSpatialEncoder` (2D CNN, time folded into batch) | `(B, 192, d)` |
| Exogenous | `(B, 192, F_exo)` clear-sky, NWP, nowcast GHI, Earth-2 hub features | MLP | `(B, 192, d)` |
| Fusion | concat `4d` → temporal Conv1d residual → MLP head | | `(B, 192, 5)` |

Monotone quantiles: `q10 = softplus(z0)`, `Δq_k = softplus(z_k)`, `q_k = cumsum` → `P10 ≤ … ≤ P90` by construction, scaled by capacity, clamped to `[0, capacity]`; solar is multiplied by `(elevation > 0)` so night output is exactly 0 MW. Training loss: pinball over `Q = (0.10, 0.25, 0.50, 0.75, 0.90)`.

## 3. Temporal cutoff alignment (leakage prevention)

Let `t0` be the forecast creation time and Δ = 15 min.

* Satellite frames: `t_k ≤ t0 − 15 min` (scan acquisition delay); the model uses the 4 frames `t0−4Δ … t0−Δ`.
* Telemetry: only `[t0 − 96Δ, t0)`; ENTSO-E gaps interpolated up to 4 MTU (1 h), never beyond.
* Nowcast: lead `k = 1..24` from the last frame, i.e. `t0 … t0 + 23Δ` (6 h).
* Day-ahead horizon: `T_future = 192` steps (48 h): `t0 … t0 + 191Δ`.
* Walk-forward split: the last training target ends before the first test `t0`; test cutoffs are 24 h apart.

## 4. Romanian prosumer split

* A75 gives *metered* utility solar (~2.85 GW) and wind onshore (>3.5 GW installed).
* ~4 GW of behind-the-meter prosumer PV is invisible in A75; it appears as reduced A65 load. RoCast therefore also carries `total_load_mw` and `net_load_ex_renewables_mw = load − solar − wind` as telemetry features.

## 5. Operational guardrails

* Satellite NaN ratio > 5 % → `ValueError` (forecast refused); ≤ 5 % → spatial-median imputation.
* pySTEPS LK is used when available; otherwise a dense NumPy Lucas-Kanade + semi-Lagrangian fallback (same shapes) is used automatically.
* Solar night clamp, capacity clamp, strict quantile monotonicity enforced in the model forward pass.
* `--mode synthetic` is fully offline and deterministic; `--mode live` needs `ROCAST_ENTSOE_API_KEY` and degrades *explicitly* (with warnings) when SEVIRI files or Earth-2 are not configured.
