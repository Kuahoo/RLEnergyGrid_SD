import numpy as np
import pytest

from sdge_microgrid.data.loader import load_week
from sdge_microgrid.data.synthetic import generate_year_traces, save_processed


def test_synthetic_bounds_and_week_length(tmp_path):
    traces = generate_year_traces(seed=0)
    assert traces["pv"].shape == (8760,)
    assert traces["pv"].min() >= 0.0
    assert traces["pv"].max() <= 8.0 + 1e-6
    assert traces["pcrit"].max() <= 4.0 + 1e-6
    assert traces["pflex"].max() <= 8.0 + 1e-6
    save_processed(tmp_path, traces)

    week = load_week(tmp_path, start_hour=100, scenario="normal")
    assert len(week.pv) == 168
    assert len(week.rho) == 168
    assert week.rho.max() < 0.2

    week_h = load_week(tmp_path, start_hour=100, scenario="highrisk")
    assert week_h.rho.mean() > week.rho.mean()


def test_highrisk_week_has_sustained_warning(tmp_path):
    save_processed(tmp_path, generate_year_traces(seed=0))
    week = load_week(tmp_path, start_hour=2000, scenario="highrisk")
    assert week.rho.mean() > 0.4


def test_unknown_scenario_rejected(tmp_path):
    save_processed(tmp_path, generate_year_traces(seed=0))
    with pytest.raises(ValueError):
        load_week(tmp_path, start_hour=100, scenario="apocalypse")
