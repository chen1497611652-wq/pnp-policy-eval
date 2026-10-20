"""单杯起点适配：保留 SimpleEnv2 的控制、相机和判定，重写双杯初始化。

初始化流程改编自 every-embodied 的 mujoco_env/y_env2.py（CC BY 4.0，
Copyright Datawhale and contributors）；上游文件不做修改。
"""
from __future__ import annotations

import numpy as np


def make_single_cup_env(xml_path, *, action_type, state_type, seed, obj_init):
    """obj_init 顺序为红杯 xyz、盘子 xyz，均为沉降前的设置值。"""
    positions = np.asarray(obj_init, dtype=np.float64)
    if positions.shape != (6,) or not np.isfinite(positions).all():
        raise ValueError("Single-cup obj_init must be six finite values: red xyz, plate xyz")
    positions = positions.copy()
    # 仅在创建真实环境时导入上游依赖，普通分支测试不需要初始化 MuJoCo。
    from mujoco_env.y_env2 import SimpleEnv2
    from mujoco_env.ik import solve_ik
    from mujoco_env.transforms import rpy2r

    class DemoSingleCupEnv(SimpleEnv2):
        def reset(self, seed=None):
            self.env.reset()
            if seed is not None:
                np.random.seed(seed)
            names = set(self.env.get_body_names(prefix="body_obj_"))
            if names != {"body_obj_mug_5", "body_obj_plate_11"}:
                raise ValueError(f"Expected only red mug and plate, got {sorted(names)}")
            # 机器人回到与双杯评测相同的 OMY home，不强行瞬移到历史首帧。
            q_zero, _, _ = solve_ik(
                env=self.env, joint_names_for_ik=self.joint_names, body_name_trgt="tcp_link",
                q_init=np.deg2rad([0, 0, 0, 0, 0, 0]),
                p_trgt=np.array([.3, 0., 1.]),
                R_trgt=rpy2r(np.deg2rad([90., 0., 90.])),
            )
            self.env.forward(q=q_zero, joint_names=self.joint_names, increase_tick=False)
            for name, pos in [("body_obj_mug_5", positions[:3]), ("body_obj_plate_11", positions[3:])]:
                self.env.set_p_base_body(body_name=name, p=pos.copy())
                self.env.set_R_base_body(body_name=name, R=np.eye(3))
            self.env.forward(increase_tick=False)
            self.last_q = np.array(q_zero, copy=True)
            self.q = np.concatenate([q_zero, np.zeros(4)])
            self.p0, self.R0 = self.env.get_pR_body(body_name="tcp_link")
            self.obj_init_pose = positions.astype(np.float32)
            self.set_instruction()
            for _ in range(100):
                self.step_env()
            self.gripper_state = False
            self.past_chars = []
            self.initialization_report = {
                "mode": "single red mug, recorded demo object initialization",
                "requested_obj_init_before_settling": positions.tolist(),
                "settle_physics_steps": 100,
                "robot_home_xyz": [.3, 0., 1.],
                "robot_home_rpy_deg": [90., 0., 90.],
                "after_settle_ee_pose": np.asarray(self.get_ee_pose()).tolist(),
                "after_settle_joint_state": np.asarray(self.get_joint_state()).tolist(),
                "after_settle_objects": {name: np.asarray(self.env.get_p_body(name)).tolist() for name in sorted(names)},
                "full_historical_robot_state_restored": False,
            }
            print("DONE SINGLE-CUP DEMO INITIALIZATION", flush=True)

        def set_instruction(self, given=None):
            text = given or "Place the red mug on the plate."
            if "blue" in text.lower() or "red" not in text.lower():
                raise ValueError("Single-cup baseline supports only the red mug task")
            super().set_instruction(text)

        def get_obj_pose(self):
            return self.env.get_p_body("body_obj_mug_5"), self.env.get_p_body("body_obj_plate_11")

        def set_obj_pose(self, p_mug_red, p_plate):
            for name, pos in [("body_obj_mug_5", p_mug_red), ("body_obj_plate_11", p_plate)]:
                self.env.set_p_base_body(body_name=name, p=pos)
                self.env.set_R_base_body(body_name=name, R=np.eye(3))
            self.step_env()

    return DemoSingleCupEnv(str(xml_path), action_type=action_type, state_type=state_type, seed=seed)
