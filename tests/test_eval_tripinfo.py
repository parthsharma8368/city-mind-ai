"""parse_tripinfo must never turn a broken file into "0 trips, 0.0 s time loss"."""
from __future__ import annotations

import pytest

from src.evaluation.tripinfo import TripinfoError, parse_tripinfo

HEAD = '<?xml version="1.0" encoding="UTF-8"?>\n<!-- generated -->\n<tripinfos>\n'


def _row(vid, depart=100.0, delay=0.0, arrival=200.0, duration=100.0, length=500.0, wait=10.0, loss=30.0, extra=""):
    return (f'    <tripinfo id="{vid}" depart="{depart}" departDelay="{delay}" arrival="{arrival}" '
            f'duration="{duration}" routeLength="{length}" waitingTime="{wait}" timeLoss="{loss}" '
            f'rerouteNo="0" vType="car"{extra}/>\n')


def _write(tmp_path, body, close=True, name="t.xml"):
    p = tmp_path / name
    p.write_text(HEAD + body + ("</tripinfos>\n" if close else ""))
    return p


def test_normal_file_and_legacy_vs_nonzero_fields(tmp_path):
    p = _write(tmp_path, _row("a", loss=30, duration=100) + _row("b", loss=50, duration=200)
               + _row("h1", length=0.0, duration=1.0, loss=1.0, wait=0.0))
    s = parse_tripinfo(p)
    # legacy fields average every row, zero-length artifact included
    assert s.trips_completed == 3
    assert s.mean_trip_timeloss == pytest.approx((30 + 50 + 1) / 3)
    # new fields exclude zero-length trips
    assert s.zero_length_trips == 1
    assert s.trips_completed_nonzero == 2
    assert s.mean_trip_timeloss_nonzero == pytest.approx(40.0)
    assert s.mean_trip_duration_nonzero == pytest.approx(150.0)
    assert s.partial is False


def test_timeloss_plus_departdelay(tmp_path):
    p = _write(tmp_path, _row("a", loss=30, delay=5) + _row("b", loss=50, delay=15))
    assert parse_tripinfo(p).mean_trip_timeloss_plus_departdelay_nonzero == pytest.approx(50.0)


def test_valid_file_with_no_trips_is_zero_not_error(tmp_path):
    s = parse_tripinfo(_write(tmp_path, ""))
    assert s.trips_completed == 0 and s.partial is False


def test_missing_file_raises(tmp_path):
    with pytest.raises(TripinfoError, match="not found"):
        parse_tripinfo(tmp_path / "nope.xml")


def test_empty_file_raises(tmp_path):
    p = tmp_path / "empty.xml"
    p.write_text("")
    with pytest.raises(TripinfoError, match="empty"):
        parse_tripinfo(p)


def test_truncated_file_raises(tmp_path):
    p = _write(tmp_path, _row("a") + _row("b") + '    <tripinfo id="c" depart="1', close=False)
    with pytest.raises(TripinfoError, match="truncated"):
        parse_tripinfo(p)


def test_unclosed_root_is_truncated(tmp_path):
    # SUMO killed before writing </tripinfos>: all rows are complete but the file is not
    p = _write(tmp_path, _row("a") + _row("b"), close=False)
    with pytest.raises(TripinfoError):
        parse_tripinfo(p)


def test_truncated_file_salvaged_only_on_request(tmp_path):
    p = _write(tmp_path, _row("a", loss=10) + _row("b", loss=20) + '    <tripinfo id="c" depart="1', close=False)
    s = parse_tripinfo(p, allow_partial=True)
    assert s.partial is True
    assert s.trips_completed == 2
    assert s.mean_trip_timeloss == pytest.approx(15.0)


@pytest.mark.parametrize("attr", ["timeLoss", "duration", "waitingTime", "routeLength", "departDelay"])
def test_missing_attribute_raises(tmp_path, attr):
    row = _row("a").replace(f'{attr}=', f'x{attr}=')
    with pytest.raises(TripinfoError, match=attr):
        parse_tripinfo(_write(tmp_path, row))


def test_timeloss_is_not_waiting_time(tmp_path):
    s = parse_tripinfo(_write(tmp_path, _row("a", wait=7.0, loss=33.0)))
    assert s.mean_trip_timeloss == pytest.approx(33.0)
    assert s.mean_trip_waiting_time == pytest.approx(7.0)
