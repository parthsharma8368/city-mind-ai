"""One SUMO command builder for every controller and environment (rl-core-004)."""
from __future__ import annotations

import pytest

from src.controllers.fixed_time import FixedTimeController
from src.controllers.sumo_actuated import SumoActuatedController
from src.environment.decentralized_sumo_env import DecentralizedSumoEnv
from src.environment.sumo_args import build_sumo_command, command_signature, get_option, with_option

from .conftest import TLS_METADATA

CFG = "/x/InTAS_train.sumocfg"


def test_default_command_equals_the_historical_per_class_command():
    assert build_sumo_command("sumo", CFG, seed=7) == [
        "sumo", "--configuration-file", CFG, "--step-length", "1.0", "--no-warnings",
        "--no-step-log", "--seed", "7",
    ]


def test_all_options_in_documented_order():
    cmd = build_sumo_command(
        "sumo", CFG, step_length=0.5, seed=3, begin=25200, end=28800.0, teleport_time=300,
        route_steps=100, additional_files=["a.xml", "b.xml"], tripinfo_output="t.xml",
        extra_args=["--foo", "bar"],
    )
    assert cmd[9:] == ["--begin", "25200", "--end", "28800", "--time-to-teleport", "300",
                       "--route-steps", "100", "--additional-files", "a.xml,b.xml",
                       "--tripinfo-output", "t.xml", "--foo", "bar"]
    assert get_option(cmd, "--seed") == "3" and get_option(cmd, "--missing") is None


def test_with_option_replaces_and_signature_ignores_output_sinks_and_order():
    a = build_sumo_command("sumo", CFG, seed=1, extra_args=["--begin", "10", "--end", "20"])
    b = with_option(build_sumo_command("sumo", CFG, seed=1, tripinfo_output="o.xml"), "--end", "20")
    b = with_option(b, "--begin", "10")
    assert command_signature(a) == command_signature(b)
    assert command_signature(a) != command_signature(with_option(a, "--seed", "2"))
    assert with_option(a, "--begin", "5").count("--begin") == 1


@pytest.mark.skipif(not TLS_METADATA.exists(), reason="tls metadata artifact missing")
def test_controllers_and_envs_build_identical_simulation_commands(tmp_path):
    """Baselines and RL envs given the same inputs produce a byte-identical
    simulation command (only output sinks / the fixed-time static program differ)."""
    cfg = tmp_path / "s.sumocfg"
    cfg.write_text("<configuration/>")
    extra = ["--begin", "25200", "--end", "28800"]
    kw = {"sumo_extra_args": extra, "route_steps": 100, "teleport_time": 300}
    cmds = {
        "actuated": SumoActuatedController(**kw).build_command(str(cfg), 42),
    }
    dec = DecentralizedSumoEnv(str(cfg), str(TLS_METADATA), seed=42, **kw)
    cmds["dec"] = dec.build_command()
    cfg_resolved = str(cfg.resolve())
    expected = command_signature(cmds["actuated"])
    for name, c in cmds.items():
        c = [cfg_resolved if t == str(cfg) else t for t in c]
        assert command_signature(c) == expected, name
    assert get_option(cmds["dec"], "--route-steps") == "100"
    assert get_option(cmds["dec"], "--time-to-teleport") == "300"

    # fixed_time = same command + the static program as extra additional file
    (tmp_path / "static.add.xml").write_text("<additional/>")
    ft = FixedTimeController(static_add_file=str(tmp_path / "static.add.xml"), **kw)
    ft_sig = command_signature(ft.build_command(str(cfg), 42))
    ft_sig.pop("--additional-files")
    assert ft_sig == expected
