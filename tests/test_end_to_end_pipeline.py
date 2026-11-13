import json

import pandas as pd

from rocast.cli import main, run_backtest
from rocast.earth2.nvidia_weather_engine import NvidiaEarth2Engine
from rocast.ingestion.grid_weather_sampler import RomanianGridWeatherSampler


def test_synthetic_generators():
    s = RomanianGridWeatherSampler()
    b = s.generate_synthetic_sen_and_nwp("2025-06-02 09:00", 96, 192)
    assert len(b.history) == 96 and len(b.future_truth) == 192
    assert b.satellite.shape == (4, 4, 24, 40)
    assert b.satellite_times[-1] <= b.future_truth.index[0] - pd.Timedelta(minutes=15)
    t, df = NvidiaEarth2Engine().generate_synthetic_earth2_ensemble(b.future_truth.index)
    assert t.shape == (192, 16, 24, 40) and len(df) == 192


def test_end_to_end_pipeline(tmp_path):
    report = run_backtest(
        "synthetic", epochs=1, output_dir=tmp_path, days=8, n_test_windows=2, train_stride=12, verbose=False
    )
    for m in report["targets"].values():
        assert m["pinball_loss_mw"] >= 0 and m["crps_mw"] >= 0
    csv = tmp_path / "predico_submission.csv"
    df = pd.read_csv(csv)
    assert not df.empty
    assert list(df.columns) == ["datetime", "q10", "q25", "q50", "q75", "q90"]
    assert df["datetime"].str.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z").all()
    assert (df[["q10", "q25", "q50", "q75", "q90"]].diff(axis=1).iloc[:, 1:] >= 0).all().all()
    assert json.loads((tmp_path / "backtest_metrics.json").read_text())["targets"]


def test_cli_entrypoint(tmp_path):
    args = ["backtest", "--mode", "synthetic", "--epochs", "1", "--days", "8", "--n-test-windows", "2"]
    assert main([*args, "--train-stride", "12", "--output-dir", str(tmp_path)]) == 0
