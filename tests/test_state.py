import numpy as np
import pytest

from src.evaluate import read_state


class FakeEnv:
    def get_joint_state(self):
        return np.array([10, 11, 12, 13, 14, 15, 99])

    def get_ee_pose(self):
        return np.array([20, 21, 22, 23, 24, 25])

def test_joint_angle():
    env = FakeEnv()

    result = read_state(env, "joint_angle", 6)

    np.testing.assert_array_equal(
        result, [10, 11, 12, 13, 14, 15]
    )

def test_invalid_type():
    env = FakeEnv()

    with pytest.raises(ValueError):
        read_state(env, "abc", 6)

def test_ee_pose():
    env = FakeEnv()

    result = read_state(env, "ee_pose", 6)

    np.testing.assert_array_equal(
        result, [20, 21, 22, 23, 24, 25]
    )

def test_invalid_slice():
    env = FakeEnv()

    # 不能取 0 项
    with pytest.raises(ValueError):
        read_state(env, "ee_pose", 0)

    # 不能取负数项
    with pytest.raises(ValueError):
        read_state(env, "ee_pose", -1)

    # 末端位姿只有 6 项，不能要求取 7 项
    with pytest.raises(ValueError):
        read_state(env, "ee_pose", 7)