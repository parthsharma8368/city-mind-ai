"""Statistics layer: paired-by-seed comparisons, CIs, MDE, improvement_pct edge cases."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.evaluation import statistics as st


def _df(values: dict[str, list[float]], metric="lag", seeds=(42, 123, 777), demand="1.0x"):
    rows = []
    for ctrl, vals in values.items():
        for s, v in zip(seeds, vals):
            rows.append({"controller": ctrl, "seed": s, "demand_id": demand, metric: v})
    return pd.DataFrame(rows)


def test_improvement_pct_sign_lower_is_better():
    assert st.improvement_pct(100.0, 80.0) == pytest.approx(20.0)      # better -> positive
    assert st.improvement_pct(100.0, 120.0) == pytest.approx(-20.0)    # worse -> negative
    assert st.improvement_pct(100.0, 120.0, lower_is_better=False) == pytest.approx(20.0)


@pytest.mark.parametrize("base", [0.0, float("nan"), float("inf")])
def test_improvement_pct_never_inf(base):
    out = st.improvement_pct(base, 5.0)
    assert math.isnan(out)


def test_compute_improvement_zero_baseline_is_nan_not_inf():
    df = _df({"base": [0, 0, 0], "tgt": [1, 2, 3]})
    out = st.compute_improvement_over_baseline(df, baseline="base", target="tgt", metric="lag")
    assert out["improvement_pct"].isna().all()
    assert not np.isinf(out["improvement_pct"]).any()


def test_compute_improvement_missing_controller_or_metric_raises():
    df = _df({"base": [1, 2, 3], "tgt": [1, 2, 3]})
    with pytest.raises(ValueError, match="baseline"):
        st.compute_improvement_over_baseline(df, baseline="nope", target="tgt", metric="lag")
    with pytest.raises(ValueError, match="target"):
        st.compute_improvement_over_baseline(df, baseline="base", target="nope", metric="lag")
    with pytest.raises(ValueError, match="metric"):
        st.compute_improvement_over_baseline(df, baseline="base", target="tgt", metric="zzz")


def test_aggregate_single_seed_std_is_nan():
    df = pd.DataFrame([{"controller": "a", "seed": 42, "demand_id": "1.0x", "backlog": 5.0}])
    agg = st.aggregate_by_controller(df)
    assert agg["backlog_mean"].iloc[0] == 5.0
    assert math.isnan(agg["backlog_std"].iloc[0])


def test_t_ci_known_value():
    lo, hi, n = st.t_confidence_interval([10.0, 12.0, 14.0])
    assert n == 3
    # mean 12, sd 2, se 2/sqrt(3), t(0.975, df=2)=4.303
    half = 4.303 * 2 / math.sqrt(3)
    assert lo == pytest.approx(12 - half, abs=0.01) and hi == pytest.approx(12 + half, abs=0.01)


def test_t_ci_n1_is_nan():
    lo, hi, n = st.t_confidence_interval([3.0])
    assert n == 1 and math.isnan(lo) and math.isnan(hi)


def test_bootstrap_flags_small_n_unreliable():
    _, _, reliable3 = st.bootstrap_ci([1.0, 2.0, 3.0])
    _, _, reliable10 = st.bootstrap_ci(list(range(10)))
    assert reliable3 is False and reliable10 is True


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    d = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    a = st.bootstrap_ci(d, seed=1)
    assert a == st.bootstrap_ci(d, seed=1)
    assert a[0] <= np.mean(d) <= a[1]


def test_mde_scales_with_sd_and_shrinks_with_n():
    m3 = st.minimum_detectable_effect(10.0, 3)
    assert m3 == pytest.approx(2 * st.minimum_detectable_effect(5.0, 3))
    assert st.minimum_detectable_effect(10.0, 10) < m3
    # n=3, df=2: power-80% MDE is large (about 4 SDs of the paired difference / sqrt(3) * ~3.9)
    assert m3 > 10.0 / math.sqrt(3) * 3.0
    assert math.isnan(st.minimum_detectable_effect(10.0, 1))


def test_paired_comparison_is_paired_by_seed_and_has_no_pvalue():
    # a big shared seed effect (+100 on seed 123) must cancel in the paired difference
    df = _df({"base": [100.0, 200.0, 100.0], "tgt": [110.0, 211.0, 109.0]})
    out = st.paired_by_seed_comparison(df, "lag", "base")
    row = out.iloc[0]
    assert row["n_seeds"] == 3
    assert row["mean_diff"] == pytest.approx(10.0)
    assert row["per_seed_diffs"] == [10.0, 11.0, 9.0]
    assert row["sd_diff"] == pytest.approx(1.0)
    assert not any("p_value" in c or c == "p" for c in out.columns)
    assert bool(row["exceeds_mde"]) is True    # 10 >> MDE for sd 1, n 3 (~4.4)


def test_paired_comparison_within_noise_is_not_flagged():
    df = _df({"base": [100.0, 100.0, 100.0], "tgt": [95.0, 108.0, 100.0]})
    row = st.paired_by_seed_comparison(df, "lag", "base").iloc[0]
    assert bool(row["exceeds_mde"]) is False
    assert row["t_ci_lo"] < 0 < row["t_ci_hi"]


def test_paired_comparison_missing_baseline_raises():
    with pytest.raises(ValueError):
        st.paired_by_seed_comparison(_df({"a": [1, 2, 3]}), "lag", "zzz")


def test_print_comparison_table_handles_single_seed(capsys):
    df = pd.DataFrame([{"controller": "a", "seed": 42, "demand_id": "1.0x", "trips_completed": 10,
                        "mean_trip_timeloss": 3.0, "backlog": 1, "total_teleports": 0}])
    st.print_comparison_table(df)
    out = capsys.readouterr().out
    assert "n/a" in out and "no significance implied" in out
