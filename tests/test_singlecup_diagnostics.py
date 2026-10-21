"""诊断不得误改观测，也不得靠压缩时间轴改变动作块含义。"""
import numpy as np
import pytest
import torch

from tools.debug_singlecup import ReplayPolicy, YawMeanProbe
from tools.finetune_singlecup import last_observation_indices


def test_keep_last_handles_nonconsecutive_duplicates_without_reordering():
    states = np.array([[1., 2.], [3., 4.], [1., 2.], [1., 2.], [3., 4.]])
    # Index 3 has a different image, so it must not be merged with index 2.
    selected = last_observation_indices(states, [b"a", b"b", b"a", b"c", b"b"])
    assert selected == [2, 3, 4]
    # The selected indices are ORIGINAL starts: i:i+chunk, not selected[i:i+chunk].
    original_actions = np.arange(7)
    np.testing.assert_array_equal(original_actions[selected[0]:selected[0]+3], [2, 3, 4])


def test_keep_last_rejects_missing_images():
    with pytest.raises(ValueError):
        last_observation_indices([[0], [1]], [b"one"])


def test_yaw_probe_changes_only_forwarded_yaw_and_resets_wrapped_policy():
    class Spy:
        config = object()

        def reset(self):
            self.resets = getattr(self, "resets", 0) + 1

        def select_action(self, batch):
            self.received = batch
            return torch.zeros((1, 7))

    spy = Spy()
    probe = YawMeanProbe(spy, 1.5)
    state = torch.tensor([[.3, .1, 1., 1.6, .01, 1.57]])
    before = state.clone()
    image = torch.zeros((1, 3, 2, 2))
    probe.reset()
    probe.select_action({"observation.state": state, "observation.image": image})
    assert spy.resets == 1
    torch.testing.assert_close(state, before)
    torch.testing.assert_close(spy.received["observation.state"][..., :5], before[..., :5])
    assert spy.received["observation.state"][0, 5] == 1.5
    assert spy.received["observation.image"] is image
    assert probe.last_state[0][5] == 1.5
    probe.reset()
    assert probe.last_state is None
    assert spy.resets == 2


def test_replay_ignores_observation_and_holds_final_target_then_resets():
    actions = np.arange(14, dtype=np.float32).reshape(2, 7)
    replay = ReplayPolicy(actions, "cpu")
    np.testing.assert_array_equal(replay.select_action({}).numpy()[0], actions[0])
    np.testing.assert_array_equal(replay.select_action({"unrelated": 99}).numpy()[0], actions[1])
    np.testing.assert_array_equal(replay.select_action({}).numpy()[0], actions[1])
    replay.reset()
    np.testing.assert_array_equal(replay.select_action({}).numpy()[0], actions[0])
