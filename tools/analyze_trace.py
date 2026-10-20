"""分析 trace_rollout 记录，导出有单位的窗口统计和诊断曲线。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    out = args.run_dir
    traces = list((out / "act_closed_loop").glob("trajectory_seed*.jsonl"))
    if len(traces) != 1:
        raise ValueError("Expected one completed trajectory")
    rows = [json.loads(s) for s in traces[0].read_text(encoding="utf-8").splitlines() if s.strip()]
    result = json.loads((out / "act_closed_loop/results.jsonl").read_text(encoding="utf-8").strip())
    assert len(rows) == result["action_steps"]
    assert [r["step"] for r in rows] == list(range(len(rows)))
    t = np.array([r["observed_elapsed_s"] for r in rows])
    assert np.all(np.diff(t) > 0)
    np.testing.assert_allclose(np.diff(t), 0.05, rtol=0, atol=1e-8)
    ee = np.array([r["ee_pose"] for r in rows])
    state = np.array([r["state"] for r in rows])
    joint = np.array([r["joint_state"][:6] for r in rows])
    held = np.array([r["held_target"][:6] for r in rows])
    action = np.array([r["action_sent"] for r in rows])
    raw = np.array([r["action_raw"] for r in rows])
    target = np.array([r["target_pos"] for r in rows])
    plate = np.array([r["plate_pos"] for r in rows])
    gripper = np.array([r["gripper_qpos"] for r in rows])
    np.testing.assert_array_equal(state, ee)
    np.testing.assert_array_equal(held[1:], action[:-1, :6])
    assert np.isfinite(np.concatenate([ee, joint, action, raw, held, target, plate], axis=1)).all()
    delta = np.vstack([np.zeros((1, 6)), np.diff(action[:, :6], axis=0)])
    delta_max_deg = np.rad2deg(np.max(np.abs(delta), axis=1))
    tracking_deg = np.rad2deg(np.sqrt(np.mean((held - joint)**2, axis=1)))
    cup_plate = np.linalg.norm(target[:, :2] - plate[:, :2], axis=1)
    tcp_cup = np.linalg.norm(ee[:, :3] - target, axis=1)
    cup_lift = (target[:, 2] - target[0, 2]) * 1000

    def window_stats(start, end):
        idx = np.flatnonzero((t >= start - 1e-7) & (t < end - 1e-7))
        if len(idx) == 0:
            return None
        pos = ee[idx, :3]
        a = action[idx, :6]
        da = np.diff(a, axis=0)
        return {
            "window_s": f"{start:g}-{end:g}", "samples": len(idx),
            "tcp_z_min_cm": float(pos[:, 2].min() * 100),
            "tcp_z_max_cm": float(pos[:, 2].max() * 100),
            "tcp_z_range_mm": float(np.ptp(pos[:, 2]) * 1000),
            "tcp_z_net_mm": float((pos[-1, 2] - pos[0, 2]) * 1000),
            "tcp_xyz_path_mm": float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum() * 1000),
            "tcp_xyz_net_mm": float(np.linalg.norm(pos[-1] - pos[0]) * 1000),
            "tcp_xyz_bbox_diagonal_mm": float(np.linalg.norm(np.ptp(pos, axis=0)) * 1000),
            "joint_range_max_deg": float(np.rad2deg(np.ptp(joint[idx], axis=0)).max()),
            "action_range_max_deg": float(np.rad2deg(np.ptp(a, axis=0)).max()),
            "action_step_change_max_deg": float(np.rad2deg(np.abs(da).max())) if da.size else 0.,
            "action_step_change_rms_deg": float(np.rad2deg(np.sqrt(np.mean(da**2)))) if da.size else 0.,
            "held_target_tracking_rms_deg": float(np.sqrt(np.mean(tracking_deg[idx]**2))),
            "gripper_cmd_min": float(action[idx, 6].min()), "gripper_cmd_max": float(action[idx, 6].max()),
            "gripper_qpos_min_rad": float(gripper[idx].min()), "gripper_qpos_max_rad": float(gripper[idx].max()),
            "red_cup_z_range_mm": float(np.ptp(target[idx, 2]) * 1000),
            "red_cup_net_displacement_mm": float(np.linalg.norm(target[idx[-1]] - target[idx[0]]) * 1000),
            "cup_plate_xy_start_cm": float(cup_plate[idx[0]] * 100),
            "cup_plate_xy_end_cm": float(cup_plate[idx[-1]] * 100),
            "tcp_cup_distance_start_cm": float(tcp_cup[idx[0]] * 100),
            "tcp_cup_distance_end_cm": float(tcp_cup[idx[-1]] * 100),
        }

    windows = [window_stats(a, b) for a, b in [(0, 15), (15, 30), (30, 60), (60, 90), (90, 120), (15, 120), (0, 120)]]
    ten_second = [window_stats(i, i + 10) for i in range(0, 120, 10)]
    summary = {"trace_rows": len(rows), "first_sim_time_s": rows[0]["sim_time_s"],
               "last_sim_time_s": rows[-1]["sim_time_s"], "observed_span_s": float(t[-1]),
               "all_recorded_values_finite": True, "state_equals_ee_pose": True,
               "held_target_matches_previous_sent_action": True,
               "max_target_lift_mm": result["max_target_lift"] * 1000,
               "gripper_raw_range": [float(raw[:, 6].min()), float(raw[:, 6].max())],
               "gripper_sent_range": [float(action[:, 6].min()), float(action[:, 6].max())],
               "windows": windows, "ten_second_windows": ten_second,
               "result": result,
               "notes": ["Rows pair current observation with next command; tracking uses previous held target.",
                         "No universal stuck threshold is imposed. Inspect net/range/path and task progress together.",
                         "ACT queues 10 actions at 20Hz, so action variation need not mean a fresh network query each tick."]}
    (out / "trajectory_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, data in [("window_summary.csv", windows), ("ten_second_summary.csv", ten_second)]:
        with (out / name).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    headers = ["time_s", "tcp_x_m", "tcp_y_m", "tcp_z_m", "tcp_roll_rad", "tcp_pitch_rad", "tcp_yaw_rad"]
    headers += [f"joint{i+1}_actual_rad" for i in range(6)]
    headers += [f"joint{i+1}_target_rad" for i in range(6)]
    headers += ["gripper_cmd_raw", "gripper_cmd_sent", "gripper_actual_rad", "max_action_step_change_deg", "held_target_tracking_rms_deg", "red_cup_z_m", "cup_plate_xy_m"]
    with (out / "trajectory.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f); writer.writerow(headers)
        for i in range(len(rows)):
            writer.writerow([t[i], *ee[i], *joint[i], *action[i, :6], raw[i, 6], action[i, 6], gripper[i], delta_max_deg[i], tracking_deg[i], target[i, 2], cup_plate[i]])

    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180})
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 10.5), sharex=True, layout="constrained")
    fig.suptitle("ACT 双杯 seed 6：延长到 120 秒后发生了什么？", fontsize=16, weight="bold")
    axes[0].plot(t, ee[:, 2] * 100, color="#0072B2", lw=1.6)
    zmid = (ee[:, 2].min() + ee[:, 2].max()) * 50
    halfspan = max(1., np.ptp(ee[:, 2]) * 60)
    axes[0].set_ylim(zmid-halfspan, zmid+halfspan)
    axes[0].set_ylabel("末端高度（厘米）")
    axes[0].set_title("① 看实际位置：是否持续下降？", loc="left", fontsize=12)
    axes[0].text(.99, .88, f"全程 {ee[:, 2].min()*100:.3f}—{ee[:, 2].max()*100:.3f} 厘米",
                 ha="right", transform=axes[0].transAxes)
    axes[1].plot(t, delta_max_deg, color="#D55E00", lw=.6)
    axes[1].set_ylabel("相邻目标最大变化（度）")
    axes[1].set_title("② 看指令：有变化，是否带来实际进展？", loc="left", fontsize=12)
    axes[2].plot(t, action[:, 6], color="#009E73", label="发出的夹爪命令（0＝张开）", lw=1.5)
    axes[2].set_ylim(-.05, 1.05)
    axes[2].set_ylabel("夹爪命令（0—1）")
    axr = axes[2].twinx()
    axr.plot(t, gripper, color="#CC79A7", ls="--", label="实际夹爪关节角", lw=1)
    axr.set_ylim(-.05, 1.1); axr.set_ylabel("实际角度（弧度）")
    axr.spines['right'].set_visible(True)
    axes[2].set_title("③ 看夹爪：有没有闭合？", loc="left", fontsize=12)
    lines = axes[2].get_lines() + axr.get_lines()
    axes[2].legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=9)
    axes[3].plot(t, cup_lift, color="#0072B2", lw=1.2, label="红杯相对首帧的高度变化")
    axes[3].axhline(30, color="#D55E00", ls="--", lw=1, label="项目抬升判据：30 毫米")
    axes[3].set_ylim(-2, max(35, cup_lift.max()+2))
    axes[3].set_ylabel("红杯抬升（毫米）")
    axes[3].set_title("④ 看任务结果：杯子有没有被抬起？", loc="left", fontsize=12)
    axes[3].legend(loc="upper right", fontsize=9)
    axes[3].set_xlabel("从首条观测起的仿真时间（秒）")
    for ax in axes:
        ax.axvline(15, color="#666666", ls=":", lw=1)
        ax.set_xlim(0, 120); ax.grid(alpha=.18)
        ax.set_xticks(np.arange(0, 121, 15))
    axes[0].text(15.8, axes[0].get_ylim()[0]+.12, "此前试跑到这里", fontsize=9, color="#555555")
    fig.savefig(out / "trajectory_overview.png")
    fig.savefig(out / "trajectory_overview.svg")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6), sharex=True, layout="constrained")
    for j in range(6):
        axes[0].plot(t, np.rad2deg(action[:, j]), lw=.8, label=f"关节 {j+1}")
    axes[0].set_ylabel("关节目标角（度）"); axes[0].legend(ncol=3, fontsize=9)
    axes[0].set_title("六个绝对关节目标与实际跟随误差")
    axes[1].plot(t, tracking_deg, lw=.8, color="#D55E00")
    axes[1].set_ylabel("上一条目标与实际角\nRMS 误差（度）")
    axes[1].set_xlabel("仿真时间（秒）")
    for ax in axes:
        ax.set_xlim(0, 120); ax.grid(alpha=.18); ax.axvline(15, color="#666666", ls=":", lw=1)
    fig.savefig(out / "joint_tracking.png"); plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items() if k not in ["ten_second_windows", "notes"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
