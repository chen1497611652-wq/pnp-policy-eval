"""Exercise the single-cup adapter without importing MuJoCo or opening a viewer."""
import sys
import types

import numpy as np
import pytest

from src.single_cup_env import make_single_cup_env


@pytest.fixture
def upstream(monkeypatch):
    class Parser:
        body_names = {"body_obj_mug_5", "body_obj_plate_11"}

        def __init__(self):
            self.positions = {}
            self.events = []
            self.physics_steps = 0

        def reset(self):
            self.events.append("reset")
            self.physics_steps = 0

        def get_body_names(self, prefix):
            assert prefix == "body_obj_"
            return sorted(self.body_names)

        def forward(self, **kwargs):
            self.events.append(("forward", kwargs))

        def set_p_base_body(self, body_name, p):
            assert body_name in self.body_names and body_name != "body_obj_mug_6"
            self.positions[body_name] = np.array(p, copy=True)
            self.events.append(("place", body_name, np.array(p, copy=True)))

        def set_R_base_body(self, body_name, R):
            assert body_name in self.body_names and body_name != "body_obj_mug_6"
            np.testing.assert_array_equal(R, np.eye(3))

        def get_pR_body(self, body_name):
            assert body_name == "tcp_link"
            return np.array([0.3, 0, 1.0]), np.eye(3)

        def get_p_body(self, name):
            assert name != "body_obj_mug_6"
            return self.positions[name].copy()

    class SimpleEnv2:
        def __init__(self, xml_path, action_type, state_type, seed):
            self.env = Parser()
            self.xml_path = xml_path
            self.action_type = action_type
            self.state_type = state_type
            self.joint_names = [f"joint{i}" for i in range(1, 7)]
            self.reset(seed)

        def step_env(self):
            # All placements must be finished before physical settling starts.
            assert set(self.env.positions) == {"body_obj_mug_5", "body_obj_plate_11"}
            self.env.physics_steps += 1
            self.env.events.append("physics")

        def set_instruction(self, given):
            if "red" not in given:
                raise ValueError("upstream red instruction expected")
            self.instruction = given
            self.obj_target = "body_obj_mug_5"

        def get_ee_pose(self):
            return np.array([0.3, 0, 1, 0.1, 0.2, 0.3])

        def get_joint_state(self):
            return np.arange(7, dtype=np.float32)

        def step(self, action):
            return "original controller", action

        def grab_image(self):
            return "original camera"

        def check_success(self):
            return "original success check"

    ik_calls = []
    q_home = np.arange(6, dtype=float) / 10

    def solve_ik(**kwargs):
        ik_calls.append(kwargs)
        return q_home.copy(), None, None

    package = types.ModuleType("mujoco_env")
    package.__path__ = []
    env_module = types.ModuleType("mujoco_env.y_env2")
    env_module.SimpleEnv2 = SimpleEnv2
    ik_module = types.ModuleType("mujoco_env.ik")
    ik_module.solve_ik = solve_ik
    transforms = types.ModuleType("mujoco_env.transforms")
    transforms.rpy2r = lambda value: np.eye(3)
    for name, module in [("mujoco_env", package), (env_module.__name__, env_module),
                         (ik_module.__name__, ik_module), (transforms.__name__, transforms)]:
        monkeypatch.setitem(sys.modules, name, module)
    return types.SimpleNamespace(Parser=Parser, base=SimpleEnv2, ik_calls=ik_calls, q_home=q_home)


def create(obj_init=None):
    if obj_init is None:
        obj_init = [0.25, 0.12, 0.82, 0.39, -0.15, 0.82]
    return make_single_cup_env("single-cup.xml", action_type="joint_angle",
                               state_type="ee_pose", seed=6, obj_init=obj_init)


def test_recorded_objects_are_placed_before_settling_and_controller_is_inherited(upstream):
    initial = np.array([0.25, 0.12, 0.82, 0.39, -0.15, 0.82])
    env = create(initial)
    parser = env.env

    assert env.obj_target == "body_obj_mug_5"
    assert parser.physics_steps == 100
    placements = [event for event in parser.events if isinstance(event, tuple) and event[0] == "place"]
    assert [event[1] for event in placements] == ["body_obj_mug_5", "body_obj_plate_11"]
    np.testing.assert_array_equal(placements[0][2], initial[:3])
    np.testing.assert_array_equal(placements[1][2], initial[3:])
    np.testing.assert_array_equal(env.q, np.r_[upstream.q_home, np.zeros(4)])
    np.testing.assert_array_equal(upstream.ik_calls[0]["p_trgt"], [0.3, 0, 1])
    assert upstream.ik_calls[0]["body_name_trgt"] == "tcp_link"
    assert env.step.__func__ is upstream.base.step
    assert env.grab_image.__func__ is upstream.base.grab_image
    assert env.check_success.__func__ is upstream.base.check_success
    assert env.initialization_report["full_historical_robot_state_restored"] is False
    assert env.initialization_report["settle_physics_steps"] == 100
    # A caller mutating its array must not change future reset placement.
    initial[:] = -99
    env.reset(6)
    np.testing.assert_array_equal(env.get_obj_pose()[0], [0.25, 0.12, 0.82])
    np.testing.assert_array_equal(env.get_obj_pose()[1], [0.39, -0.15, 0.82])
    assert parser.physics_steps == 100


@pytest.mark.parametrize("names", [
    {"body_obj_mug_5", "body_obj_mug_6", "body_obj_plate_11"},
    {"body_obj_mug_5"},
])
def test_wrong_scene_fails_before_ik_or_physics(upstream, names):
    upstream.Parser.body_names = names
    with pytest.raises(ValueError, match="Expected only red mug and plate"):
        create()
    assert upstream.ik_calls == []


@pytest.mark.parametrize("task", ["Place the blue mug on the plate.", "Pick an object."])
def test_unsupported_task_is_rejected(upstream, task):
    env = create()
    with pytest.raises(ValueError, match="supports only the red mug task"):
        env.set_instruction(task)
    assert env.obj_target == "body_obj_mug_5"


@pytest.mark.parametrize("values", [[1, 2, 3], [1, 2, 3, 4, 5, float("nan")]])
def test_invalid_recorded_positions_fail_before_upstream_creation(values):
    with pytest.raises(ValueError, match="six finite values"):
        create(values)
