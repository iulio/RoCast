"""Predico-style probabilistic evaluation and submission export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rocast.models.pvnet_romania import QUANTILES

SUBMISSION_COLUMNS = ["datetime", "q10", "q25", "q50", "q75", "q90"]


class PredicoEvaluator:
    def __init__(self, quantiles: tuple[float, ...] = QUANTILES) -> None:
        self.quantiles = np.asarray(quantiles, dtype=np.float64)

    def pinball_loss_mw(self, y_true: np.ndarray, y_quantiles: np.ndarray) -> float:
        y = np.asarray(y_true, dtype=np.float64).reshape(-1, 1)
        q = np.asarray(y_quantiles, dtype=np.float64).reshape(-1, len(self.quantiles))
        err = y - q
        return float(np.maximum(self.quantiles * err, (self.quantiles - 1.0) * err).mean())

    def compute_metrics(
        self, y_true: np.ndarray, y_quantiles: np.ndarray, nwp_baseline: np.ndarray | None = None
    ) -> dict[str, float]:
        y = np.asarray(y_true, dtype=np.float64).reshape(-1)
        q = np.asarray(y_quantiles, dtype=np.float64).reshape(-1, len(self.quantiles))
        pin = self.pinball_loss_mw(y, q)
        p50 = q[:, len(self.quantiles) // 2]
        mae = float(np.abs(y - p50).mean())
        metrics = {
            "pinball_loss_mw": pin,
            "crps_mw": 2.0 * pin,
            "mae_p50_mw": mae,
            "rmse_p50_mw": float(np.sqrt(np.mean((y - p50) ** 2))),
            "p10_p90_coverage_pct": float(100.0 * np.mean((y >= q[:, 0]) & (y <= q[:, -1]))),
        }
        if nwp_baseline is not None:
            mae_nwp = float(np.abs(y - np.asarray(nwp_baseline, dtype=np.float64).reshape(-1)).mean())
            metrics["mae_nwp_mw"] = mae_nwp
            metrics["skill_score_vs_nwp_pct"] = float(100.0 * (1.0 - mae / mae_nwp)) if mae_nwp > 0 else 0.0
        else:
            metrics["skill_score_vs_nwp_pct"] = 0.0
        return metrics

    @staticmethod
    def export_predico_submission(timestamps_utc, y_quantiles: np.ndarray, output_csv: str | Path) -> Path:
        idx = pd.DatetimeIndex(timestamps_utc)
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        q = np.asarray(y_quantiles, dtype=np.float64).reshape(len(idx), -1)
        df = pd.DataFrame(q, columns=SUBMISSION_COLUMNS[1:])
        df.insert(0, "datetime", idx.strftime("%Y-%m-%dT%H:%M:%SZ"))
        path = Path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, float_format="%.3f")
        return path

    @staticmethod
    def save_report(report: dict, output_json: str | Path) -> Path:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return path
