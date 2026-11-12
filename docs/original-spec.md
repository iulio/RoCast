
================================================================================
0. AUTONOMOUS AGENT EXECUTION DIRECTIVES (ZERO HUMAN-IN-THE-LOOP)
================================================================================
1. **NO PAUSING OR QUESTIONS:** Do NOT ask for confirmation, clarification, or permission at any stage. Make senior-level architectural decisions autonomously and execute from start to finish in a single continuous workflow.
2. **ZERO PLACEHOLDERS:** Never write `pass`, `TODO`, `NotImplementedError`, or truncated functions. Every file must be 100% implemented, type-annotated (Python 3.11+), and runnable.
3. **ZERO GENERATIVE LLMs/VLMs IN THE FORECASTING LOOP:** Do not use any text or vision-language LLMs in the pipeline. Use strictly:
   - Deterministic radiative transfer & solar geometry (`pvlib` Ineichen clear-sky + Heliosat cloud albedo index)
   - Deterministic Lagrangian satellite optical flow (`pysteps` Lucas-Kanade `"LK"` + semi-Lagrangian advection)
   - **NVIDIA Earth-2** prognostic weather ensembles (`earth2studio` FourCastNet 3 `FCN3` / `StormScope` / `CorrDiff` + NVIDIA Earth-2 NIM REST `/v1/infer` `.tar` stream client)
   - **OpenClimateFix `PVNet`-style** PyTorch 3D Spatio-Temporal Fusion Network (`PVNetRomaniaFusion`) + optional Time Series Foundation Model (`Chronos-Bolt`) prior injection, trained with calibrated Pinball Loss and strict quantile monotonicity.
4. **DUAL-MODE EXECUTION (`live` vs. `synthetic`):**
   - Implement the complete `live` ingestion clients for ENTSO-E (`entsoe-py`), Open-Meteo NWP (`requests`), EUMETSAT (`satpy`/`xarray`), and NVIDIA `earth2studio` / NIM (`/v1/infer`).
   - Also implement a deterministic, physically realistic `--mode synthetic` tensor & time-series generator inside the data/weather modules so the entire pipeline, unit tests, walk-forward backtester, Predico CSV exporter, and LinkedIn chart generator execute 100% offline on CPU or local GPU without failing on missing API keys or external network calls.
5. **SELF-HEALING LOOP:** After writing all files, create/activate the local Python environment, install dependencies, run the `pytest` suite, run the CLI end-to-end pipeline, and generate the LinkedIn visual. If any import, tensor shape, or runtime error occurs, diagnose and fix it autonomously until every verification command exits with code `0`.

================================================================================
1. REPOSITORY DIRECTORY STRUCTURE TO CREATE
================================================================================
Create the following exact repository layout in the current workspace:

.
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
├── PLAN.md
├── assets/
│   └── (auto-generated rocast_linkedin_hero.png)
├── outputs/
│   └── (auto-generated predico_submission.csv & backtest_metrics.json)
├── rocast/
│   ├── __init__.py
│   ├── config.py
│   ├── cli.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── grid_weather_sampler.py
│   ├── physics/
│   │   ├── __init__.py
│   │   └── clearsky_advection.py
│   ├── earth2/
│   │   ├── __init__.py
│   │   └── nvidia_weather_engine.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── pvnet_romania.py
│   └── backtest/
│       ├── __init__.py
│       └── predico_evaluator.py
├── scripts/
│   └── generate_linkedin_hero.py
└── tests/
    ├── __init__.py
    ├── test_physics_advection.py
    ├── test_pvnet_fusion.py
    └── test_end_to_end_pipeline.py

================================================================================
2. DOMAIN & MATHEMATICAL SPECIFICATIONS (ROMANIA SEN GRID)
================================================================================
### 2.1 The Romanian Prosumer Split & Spatial Domain
- **Bidding Zone:** `10YRO-TEL------P` (`Europe/Bucharest`, 15-minute MTU resolution).
- **Dual-Target Formulation:** Romania has >6.7 GW of solar capacity (~4.05 GW behind-the-meter prosumers vs. ~2.85 GW utility-scale PV) and >3.55 GW of onshore wind. The pipeline must model:
  1. Metered Utility Solar (`solar_mw`) and Onshore Wind (`wind_onshore_mw`) from ENTSO-E Document `A75`.
  2. Net System Load (`total_load_mw` and `net_load_ex_renewables_mw = total_load_mw - solar_mw - wind_onshore_mw`) from ENTSO-E Document `A65` to capture behind-the-meter prosumer self-consumption.
- **Romania Bounding Box:** `Lon: 20.0°E to 30.0°E`, `Lat: 43.5°N to 48.5°N` (captures Black Sea marine boundary layer and Carpathian orographic lifting).
- **5 Core Renewable Generation Hubs (defined in `rocast/config.py`):**
  1. `dobrogea_wind`: Lat `44.50`, Lon `28.30`, Alt `85m`, Capacity `2650.0 MW` (Constanța/Tulcea)
  2. `moldova_wind`: Lat `45.80`, Lon `27.80`, Alt `140m`, Capacity `750.0 MW` (Focșani Gate / Galați / Vaslui / Vrancea corridor)
  3. `wallachia_solar`: Lat `44.30`, Lon `25.50`, Alt `90m`, Capacity `1400.0 MW` (Muntenia)
  4. `oltenia_solar`: Lat `44.10`, Lon `23.80`, Alt `115m`, Capacity `850.0 MW`
  5. `transylvania_solar`: Lat `46.50`, Lon `24.50`, Alt `340m`, Capacity `600.0 MW`

### 2.2 Temporal Cutoff & Leakage Prevention ($t_0$ Gate)
All features must be strictly aligned relative to forecast creation cutoff $t_0$:
- **EUMETSAT Satellite Lag:** Account for 15-minute scan acquisition delay ($t \le t_0 - 15\text{min}$).
- **ENTSO-E Telemetry Latency:** Account for publication lag and linearly interpolate gaps up to 4 MTUs (1 hour).
- **Horizon:**
  - Intraday Nowcast Horizon: $T_{nowcast} = 24$ steps (6 hours at 15-min MTU).
  - Day-Ahead Competition Horizon: $T_{future} = 192$ steps (48 hours at 15-min MTU).
- **Quantiles:** $Q = (0.10, 0.25, 0.50, 0.75, 0.90)$.

================================================================================
3. MODULE-BY-MODULE IMPLEMENTATION SPECIFICATIONS
================================================================================

### Module 1: `rocast/config.py`
- Use `pydantic` and `pydantic-settings` (`BaseSettings`).
- Define `HubCoordinate` and `RoCastSettings` with all Romanian hub coordinates, bounding box `(20.0, 43.5, 30.0, 48.5)`, spatial crop shape `(24, 40)`, EUMETSAT channels `["VIS006", "VIS008", "IR_108", "WV_062"]`, Earth-2 variables `["u10m", "v10m", "u100m", "v100m", "t2m", "sp", "tcwv", "z500"]`, `max_satellite_nan_ratio = 0.05`, and `quantiles = (0.10, 0.25, 0.50, 0.75, 0.90)`.

### Module 2: `rocast/physics/clearsky_advection.py`
1. **`ClearSkyPhysicsCalculator`:**
   - Uses `pvlib.location.Location(lat, lon, tz="UTC", altitude=alt)` for each Romanian hub to compute `get_solarposition` (`apparent_elevation`, `apparent_zenith`) and `get_clearsky(timestamps, model="ineichen")` (`ghi`, `dni`, `dhi`).
   - Implements `heliosat_cloud_index(vis_reflectance, rho_ground_min, rho_cloud_max=0.85)`:
     $$n(x, y, t) = \text{clip}\left(\frac{\rho(x, y, t) - \rho_{\min}(x, y)}{\max(\rho_{\max} - \rho_{\min}(x, y), 10^{-4})}, 0.0, 1.2\right)$$
     $$K_c = \text{clip}(1.0 - 0.85 n, 0.05, 1.05)$$
2. **`PyStepsCloudNowcaster`:**
   - **Operational Validation Gate:** `validate_satellite_stack(sat_stack)` checks if `np.isnan(sat_stack).mean() > 0.05` (raises `ValueError` if exceeded; otherwise imputes minor NaNs with spatial median).
   - **Optical Flow & Advection:** Uses `pysteps.motion.get_method("LK")` (with an automatic dense NumPy Lucas-Kanade / spatial gradient optical-flow fallback if `pysteps` C-extensions are unavailable in minimal environments) and `pysteps.extrapolation.get_method("semilagrangian")` to extrapolate the last $\ge 3$ cloud-index frames $(T_{past}, H, W)$ forward for `n_lead_steps=24` (6 hours), returning `advected_cloud_index` of shape `(24, H, W)` and `velocity_field` of shape `(2, H, W)`.
   - **Hub Irradiance Projection:** Extracts $3\times 3$ pixel neighborhoods around each Romanian hub from `advected_cloud_index`, converts to $K_c$, and multiplies by `pvlib` Ineichen Clear-Sky GHI to output deterministic 0–6h GHI nowcasts.

### Module 3: `rocast/earth2/nvidia_weather_engine.py`
Implement `NvidiaEarth2Engine` with three clean methods:
1. **`run_earth2studio_fcn3_ensemble(init_times, n_steps=8)`:**
   - Uses lazy imports for `earth2studio.models.px.FCN3`, `earth2studio.data.GFS`, `earth2studio.perturbation.SphericalGaussian`, `earth2studio.io.ZarrBackend`, and `earth2studio.run.ensemble`.
   - Crops global output to Romania `[43.5°N–48.5°N, 20.0°E–30.0°E]`, computes `ws100m = hypot(u100m, v100m)` and `ws10m = hypot(u10m, v10m)`, and computes ensemble mean ($\mu$) and spread ($\sigma$, renamed `{var}_spread`) across ensemble members.
2. **`call_fourcastnet_nim(initial_state, simulation_length=8)`:**
   - Sends `initial_state` (`float32` `.npy` stream of shape `(1, 1, 72, 721, 1440)`) via `httpx.Client` multipart `POST` to `cfg.earth2_nim_url` (`/v1/infer` with `Bearer {cfg.ngc_api_key}`).
   - Unpacks the returned `.tar` stream in memory (`tarfile.open`), parses `{lead_time}_{batch_index}.npy`, and slices the global 0.25° grid to Romania indices `lat_idx [166:186], lon_idx [80:120]`.
3. **`generate_synthetic_earth2_ensemble(target_timestamps)`:**
   - Generates a physically realistic 4D spatial tensor `(T_future, C_e2=16, H=24, W=40)` (8 variables $\times$ [mean, spread]) + hub-level covariate DataFrame simulating a Black Sea / Dobrogea & Focșani Gate wind jet and Wallachia radiative field so offline backtests and unit tests run deterministically without requiring a 40GB GPU or NIM container.

### Module 4: `rocast/ingestion/grid_weather_sampler.py`
Implement `RomanianGridWeatherSampler`:
1. **`fetch_entsoe_sen_history(start_utc, end_utc)`:** Uses `EntsoePandasClient` to pull `A75` (Solar, Wind Onshore) and `A65` (Total Load) for `10YRO-TEL------P`, resamples to 15-minute MTU, interpolates up to 4 intervals, and computes `net_load_ex_renewables_mw`.
2. **`fetch_openmeteo_hub_nwp(start_date, end_date)`:** Queries Open-Meteo `https://api.open-meteo.com/v1/forecast` across all 5 Romanian hubs for `shortwave_radiation`, `direct_normal_irradiance`, `diffuse_radiation`, `cloud_cover`, `cloud_cover_low`, `wind_speed_100m`, `wind_direction_100m`, `temperature_2m`, `surface_pressure`, and computes cubic wind power proxy $\text{clip}(v_{100\text{m}}, 0, 25)^3$.
3. **`generate_synthetic_sen_and_nwp(start_utc, periods_past=96, periods_future=192)`:** Generates realistic, time-aligned 15-minute historical SEN telemetry, future NWP covariates, and EUMETSAT satellite frames `(T_past=4, C_sat=4, H=24, W=40)` for offline testing and benchmarking.

### Module 5: `rocast/models/pvnet_romania.py`
Implement the OpenClimateFix `PVNet`-inspired PyTorch architecture:
1. **`SatelliteAdvection3DEncoder(nn.Module)`:** 3D CNN (`Conv3d` -> `BatchNorm3d` -> `SiLU` -> `MaxPool3d` -> `AdaptiveAvgPool3d`) encoding the 5D satellite + advected cloud tensor `(B, C_sat, T_sat, H, W)` into `(B, d_model)`.
2. **`Earth2EnsembleSpatialEncoder(nn.Module)`:** 2D spatial CNN folded across time `(B*T_future, C_e2, H_ro, W_ro)` encoding Earth-2 ensemble mean & spread maps into `(B, T_future, d_model)`.
3. **`PVNetRomaniaFusion(nn.Module)`:**
   - Encodes historical 15-min SEN telemetry `(B, T_past, F_hist)` via a 2-layer `nn.GRU` (with an optional hook to concatenate zero-shot **Chronos-Bolt** quantile priors if installed).
   - Projects future exogenous covariates (`pvlib` clear-sky + `pysteps` hub GHI nowcasts + Open-Meteo NWP + Earth-2 hub features) `(B, T_future, F_exo)` into `(B, T_future, d_model)`.
   - Concatenates `[sat_ctx, hist_ctx, e2_seq, exo_seq]` of dimension `4 * d_model` into a residual MLP head outputting raw quantile logits `(B, T_future, 5)`.
   - **Strict Monotonicity & Physical Guardrails:**
     - Base quantile $q_{10} = \text{Softplus}(z_0)$, increments $\Delta q_k = \text{Softplus}(z_k)$, cumulative sum $q_k = q_0 + \sum_{j=1}^k \Delta q_j$ so $P_{10} \le P_{25} \le P_{50} \le P_{75} \le P_{90}$ is mathematically guaranteed everywhere.
     - Clamps output to `[0.0, capacity_ceiling_mw]`.
     - If `solar_elevation` tensor `(B, T_future)` is passed (for PV forecasting), multiplies all quantiles by `(solar_elevation > 0.0).float()` so nighttime solar irradiance is strictly `0.0 MW`.
4. **`CalibratedPinballLoss(nn.Module)`:**
   $$\mathcal{L}_{pinball}(y, \hat{y}) = \frac{1}{|Q| \cdot B \cdot T} \sum_{q \in Q} \max\left(q(y - \hat{y}_q), (q - 1)(y - \hat{y}_q)\right)$$

### Module 6: `rocast/backtest/predico_evaluator.py`
Implement `PredicoEvaluator`:
- Computes `pinball_loss_mw`, `crps_mw` ($2 \times \text{mean pinball loss}$), `mae_p50_mw`, `rmse_p50_mw`, `skill_score_vs_nwp_pct`, and `p10_p90_coverage_pct`.
- Implements `export_predico_submission(timestamps_utc, y_quantiles, output_csv)` writing columns `datetime,q10,q25,q50,q75,q90` in ISO-8601 UTC format (`%Y-%m-%dT%H:%M:%SZ`).
- Saves a summary JSON report to `outputs/backtest_metrics.json`.

### Module 7: `rocast/cli.py`
Implement a clean `argparse` or `typer` CLI supporting:
- `python -m rocast.cli backtest --mode [synthetic|live] --epochs 5 --output-dir outputs`
  - Runs the full pipeline: data sampling -> `pvlib` clear-sky + `pysteps` cloud advection -> `NvidiaEarth2Engine` tensor extraction -> `PVNetRomaniaFusion` training & inference -> `PredicoEvaluator` metrics calculation -> exports `outputs/predico_submission.csv` and `outputs/backtest_metrics.json` -> prints a formatted benchmark table to stdout.

### Module 8: `scripts/generate_linkedin_hero.py`
Implement the standalone dark-mode dual-panel Matplotlib visualization script (`240 DPI`, `18x9.5 inches`, saved to `assets/rocast_linkedin_hero.png`):
- **Header Banner:** Title & stack summary (`NVIDIA Earth2Studio (FourCastNet 3 + CorrDiff) • EUMETSAT + pySTEPS Optical Flow • pvlib Clear-Sky Physics • OpenClimateFix PVNet 3D Fusion`).
- **Panel A (Left):** Romania map `[20.0°E–30.0°E, 43.5°N–48.5°N]` with Romania's 32-vertex national border polygon, Carpathian Ridge arc, Black Sea label, Earth-2 100m wind speed heatmap (`viridis`), EUMETSAT Heliosat cloud contours, `pySTEPS` Lagrangian velocity vectors (`quiver`), and annotated markers for all 5 Romanian wind/solar hubs (including Dobrogea and Focșani Gate).
- **Panel B (Right):** 48-hour, 15-minute MTU probabilistic fan chart (`P10–P90` and `P25–P75` shaded bands, `P50` cyan curve, ENTSO-E Actuals white curve, and lagging standard 25km Day-Ahead NWP coral dashed curve), annotated with callouts for the Wallachia cloud drop and Dobrogea/Focșani Gate wind surge, plus an embedded monospace `PREDICO BACKTEST SCORECARD (RO ZONE)` box.

### Module 9: Documentation (`PLAN.md` and `README.md`)
- Generate `PLAN.md` containing the system architecture, tensor topology ASCII diagram, temporal cutoff alignment math, and operational guardrails.
- Generate a polished, badge-ready `README.md` embedding `assets/rocast_linkedin_hero.png`, explaining the Romanian prosumer split (`A75` vs `A65`), the 3-branch architecture (`pySTEPS` + `pvlib`, `NVIDIA Earth2Studio` / NIM, and `PVNetRomaniaFusion`), installation via `uv` / `pip`, CLI usage, and Predico benchmark results.

================================================================================
4. AUTOMATED VERIFICATION & DEFINITION OF DONE (MUST EXECUTE IN TERMINAL)
================================================================================
Before finishing your turn, you MUST execute the following steps in the terminal and fix any issues until all checks pass with exit code `0`:

1. **Install Dependencies:**
   Ensure `numpy`, `pandas`, `scipy`, `matplotlib`, `torch`, `pydantic`, `pydantic-settings`, `pvlib`, `xarray`, `requests`, `httpx`, `entsoe-py`, `pysteps` (or verify graceful optical-flow fallback), and `pytest` are installed in the workspace virtual environment.
2. **Run Unit & Integration Tests (`pytest -v`):**
   - `tests/test_physics_advection.py`: Verify Heliosat cloud index bounds `[0.0, 1.2]`, `pvlib` clear-sky GHI strictly `0.0` at midnight UTC in Romania, >5% NaN satellite rejection gate, and 24-step cloud advection output shape `(24, H, W)`.
   - `tests/test_pvnet_fusion.py`: Verify `PVNetRomaniaFusion` forward pass output shape `(B, 192, 5)`, strict quantile non-crossing (`(out[..., 1:] >= out[..., :-1]).all()`), capacity clamping, and nighttime solar zero-clamping.
   - `tests/test_end_to_end_pipeline.py`: Verify the full synthetic backtest pipeline produces valid CRPS/Pinball metrics and writes a non-empty `predico_submission.csv` with exact columns `["datetime", "q10", "q25", "q50", "q75", "q90"]`.
3. **Run the End-to-End CLI & Visual Generator:**
   - Execute: `python -m rocast.cli backtest --mode synthetic --epochs 3 --output-dir outputs`
   - Execute: `python scripts/generate_linkedin_hero.py`
4. **Verify Artifacts Exist on Disk:**
   Confirm `assets/rocast_linkedin_hero.png`, `outputs/predico_submission.csv`, `outputs/backtest_metrics.json`, `README.md`, and `PLAN.md` exist and are non-empty.

Start executing immediately. Do not stop until all verification commands have been run and passed.