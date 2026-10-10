from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Thresholds:
    """ 八个成功判据的阈值定义。"""
    lift_min: float = 0.03
    lift_run_min: int = 3
    upright_cos_min: float = 0.7
    target_plate_dz_max: float = 0.08
    plate_xy_disp_max: float = 0.05
    gripper_open_qpos_max: float = 0.1
    tcp_high_min: float = 0.9
    stable_steps_min: int = 5
DEFAULT_THRESHOLDS = Thresholds()

def read_metrics(env,initial_plate_pos) -> dict:

    target_pos = np.asarray(env.env.get_p_body(env.obj_target), dtype=np.float64)
    plate_pos = np.asarray(env.env.get_p_body("body_obj_plate_11"), dtype=np.float64)
    target_rotation = np.asarray( env.env.get_R_body(env.obj_target), dtype=np.float64)
    gripper_qpos = float(env.env.get_qpos_joint("rh_r1")[0])
    tcp_z = float(env.env.get_p_body("tcp_link")[2])
    legacy_success = bool(env.check_success())

    return {
        "legacy_success": legacy_success,
        "xy_dist": float(np.linalg.norm(target_pos[:2] - plate_pos[:2])),  # 只报告
        "target_z": float(target_pos[2]),
        "plate_z": float(plate_pos[2]),
        "upright_cos": float(target_rotation[2, 2]),
        "plate_xy_displacement": float( np.linalg.norm(plate_pos[:2] - initial_plate_pos[:2])),
        # 下面两个是原始读数，不是布尔 —— 判定搬到 update_metrics，
        # 否则 0.1 / 0.9 这两条边界就没法用纯函数测。
        "gripper_qpos": gripper_qpos,
        "tcp_z": tcp_z,
    }

def update_metrics(metrics: dict, *, max_target_lift: float, max_lifted_run: int, stable_place_steps: int, thresholds: Thresholds=DEFAULT_THRESHOLDS) -> tuple[dict, int]:

    target_z=metrics["target_z"]
    plate_z=metrics["plate_z"]
    gripper_open=metrics["gripper_qpos"] < thresholds.gripper_open_qpos_max
    tcp_high=metrics["tcp_z"] > thresholds.tcp_high_min
    placement_candidate = bool(
        metrics["legacy_success"]
        and max_target_lift >= thresholds.lift_min
        and max_lifted_run >= thresholds.lift_run_min
        and metrics["upright_cos"] >= thresholds.upright_cos_min
        and abs(target_z - plate_z) < thresholds.target_plate_dz_max
        and metrics["plate_xy_displacement"] < thresholds.plate_xy_disp_max
        and gripper_open
        and tcp_high
    )
    stable_place_steps = stable_place_steps + 1 if placement_candidate else 0
    physical_success = bool(placement_candidate and stable_place_steps >= thresholds.stable_steps_min)
    return {
        "legacy_success": metrics["legacy_success"],
        "physical_success": physical_success,
        "xy_dist": metrics["xy_dist"],
        "target_z": target_z,
        "plate_z": plate_z,
        "max_target_lift": float(max_target_lift),
        "max_lifted_run": int(max_lifted_run),
        "upright_cos": metrics["upright_cos"],
        "plate_xy_displacement": metrics["plate_xy_displacement"],
        "placement_candidate": placement_candidate,
        "stable_place_steps": int(stable_place_steps),
        "gripper_open": gripper_open,
        "tcp_high": tcp_high,
    }, stable_place_steps