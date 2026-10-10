import numpy as np
import pytest
from src.success import update_metrics

PASSING={
    "legacy_success": True,
    "xy_dist": 0.0,
    "target_z": 0.0,
    "plate_z": 0.0,
    "upright_cos": 1.0,
    "plate_xy_displacement": 0.0,
    "gripper_qpos": 0.0,
    "tcp_z": 1.0,
}

PASSING_PARAMS={
    "max_target_lift": 1.0,
    "max_lifted_run": 10,
    "stable_place_steps": 4,
}

EXPECTED_KEYS = {
    "legacy_success", "physical_success", "xy_dist", "target_z", "plate_z",
    "max_target_lift", "max_lifted_run", "upright_cos", "plate_xy_displacement",
    "placement_candidate", "stable_place_steps", "gripper_open", "tcp_high",
}

def judge(overrides=None, **params):
    """跑一次判定。overrides 改 metrics 里的字段，**params 覆盖状态量。"""
    values=dict(PASSING)
    if overrides:
        values.update(overrides)
    args=dict(PASSING_PARAMS)
    args.update(params)
    return update_metrics(values, **args)

def test_passing_baseline_actually_passes():
    """底座自检：PASSING 这套输入必须让 8 条判据全部通过。

    这条红了 = 问题在测试脚手架（键名/取值/judge），不在 src/success.py。
    """
    verdict, steps = judge()
    assert verdict["placement_candidate"] is True
    assert verdict["physical_success"] is True
    assert steps == 5          # 4 + 1

@pytest.mark.parametrize("value,expected", [(0.03, True), (0.029999, False)])
def test_lift_boundary(value, expected):
    verdict, _ = judge(max_target_lift=value)
    assert verdict["placement_candidate"] is expected

@pytest.mark.parametrize("value,expected", [(3, True), (2, False)])
def test_lift_run_boundary(value, expected):
    verdict, _ = judge(max_lifted_run=value)
    assert verdict["placement_candidate"] is expected

@pytest.mark.parametrize("value,expected", [(0.7, True), (0.6999, False)])
def test_upright_cos_boundary(value, expected):
    verdict, _ = judge({"upright_cos": value})
    assert verdict["placement_candidate"] is expected

@pytest.mark.parametrize("value,expected", [(0.08, False), (0.0799, True)])
def test_target_plate_dz_boundary(value, expected):
    # plate_z 保持 0.0 → dz 精确等于 value
    verdict, _ = judge({"target_z": value})
    assert verdict["placement_candidate"] is expected

@pytest.mark.parametrize("value,expected", [(0.05, False), (0.0499, True)])
def test_plate_xy_disp_boundary(value, expected):
    verdict, _ = judge({"plate_xy_displacement": value})
    assert verdict["placement_candidate"] is expected

@pytest.mark.parametrize("value,expected", [(0.1, False), (0.0999, True)])
def test_gripper_qpos_boundary(value, expected):
    verdict, _ = judge({"gripper_qpos": value})
    assert verdict["placement_candidate"] is expected

@pytest.mark.parametrize("value,expected", [(0.9, False), (0.9001, True)])
def test_tcp_z_boundary(value, expected):
    verdict, _ = judge({"tcp_z": value})
    assert verdict["placement_candidate"] is expected

def test_stable_steps_increments_by_one():
    _, steps = judge(stable_place_steps=0)
    assert steps == 1

def test_stable_steps_needs_five_consecutive():
    steps = 0
    for expected in (1, 2, 3, 4):
        verdict, steps = judge(stable_place_steps=steps)
        assert steps == expected
        assert verdict["physical_success"] is False
    # 接在上面那个 for 循环之后，同一个函数里
    verdict, steps = judge(stable_place_steps=steps)
    assert steps == 5
    assert verdict["physical_success"] is True

def test_stable_steps_resets_on_any_failure():
    _, steps = judge(stable_place_steps=0)
    assert steps == 1
    _, steps = judge(stable_place_steps=steps)
    assert steps == 2
    # 中间夹一次失败 → 必须归零，而不是停在 2
    _, steps = judge({"legacy_success": False}, stable_place_steps=steps)
    assert steps == 0

def test_key_set_is_exactly_the_thirteen():
    verdict, _ = judge()
    assert set(verdict) == EXPECTED_KEYS

def test_params_are_cast_away_from_numpy():
    """三个参数自带 cast，喂 numpy 标量进来必须出来 Python 标量。"""
    verdict, _ = update_metrics(
        dict(PASSING),
        max_target_lift=np.float64(1.0),
        max_lifted_run=np.int64(10),
        stable_place_steps=np.int64(4),
    )
    assert type(verdict["max_target_lift"]) is float
    assert type(verdict["max_lifted_run"]) is int
    assert type(verdict["stable_place_steps"]) is int

def test_passthrough_and_derived_types():
    """转交字段与两个派生布尔的类型。

    注意: xy_dist / upright_cos / plate_xy_displacement 是转交字段,
    update_metrics 不做 cast —— 它们的类型由 read_metrics 保证,
    而 read_metrics 需要真 env（本文件不测）。最终防线是 ③。
    """
    verdict, _ = judge()
    assert type(verdict["xy_dist"]) is float
    assert type(verdict["upright_cos"]) is float
    assert type(verdict["legacy_success"]) is bool
    assert type(verdict["gripper_open"]) is bool
    assert type(verdict["tcp_high"]) is bool
    assert type(verdict["placement_candidate"]) is bool

def test_legacy_success_vetoes_everything_else():
    """其余 7 条全部满足，仅 legacy_success=False → 仍然不通过。"""
    verdict, steps = judge({"legacy_success": False})
    assert verdict["placement_candidate"] is False
    assert verdict["physical_success"] is False
    assert steps == 0


