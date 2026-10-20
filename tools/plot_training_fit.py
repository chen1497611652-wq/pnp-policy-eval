"""将 check_training_fit 的原始预测画成时间对齐的图，不再调用模型。"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    out = parser.parse_args().output
    data = np.load(out/"predictions.npz")
    t, target, fresh, queued = [data[k] for k in ["times", "target", "fresh", "queued"]]
    metrics = json.loads((out/"metrics.json").read_text(encoding="utf-8"))
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 170})
    colors = {"target": "#222222", "fresh": "#0072B2", "queued": "#D55E00"}
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, layout="constrained")
    fig.suptitle("ACT 原题检查：只给训练图片和状态，动作答案单独核对", fontsize=16, weight="bold")
    for key, pred, style in [("target", target, "-"), ("fresh", fresh, "--"), ("queued", queued, ":")]:
        axes[0].plot(t, np.rad2deg(pred[:, 2]), style, color=colors[key], lw=1.2,
                     label={"target": "记录的目标动作", "fresh": "每帧重看观测", "queued": "每 10 步重看观测"}[key])
    axes[0].set_ylabel("第 3 关节目标（度）")
    axes[0].set_title("① 整体轨迹：这里展示平均误差最大的第 3 关节", loc="left", fontsize=12)
    axes[0].legend(loc="lower left", ncol=3, fontsize=9)
    for key, pred in [("fresh", fresh), ("queued", queued)]:
        e = np.rad2deg(np.abs(pred[:, :6]-target[:, :6])).mean(1)
        axes[1].plot(t, e, color=colors[key], lw=.8,
                     label=("每帧重看" if key=="fresh" else "每 10 步重看"))
    axes[1].set_ylabel("六关节平均绝对误差（度）")
    axes[1].set_title("② 局部误差：平均值不代表每个时刻都准确", loc="left", fontsize=12)
    axes[1].legend(loc="upper right", fontsize=9)
    for key, pred, style in [("target", target, "-"), ("fresh", fresh, "--"), ("queued", queued, ":")]:
        axes[2].plot(t, np.clip(pred[:, 6], 0, 1), style, color=colors[key], lw=1.5,
                     label={"target": "记录的夹爪命令", "fresh": "每帧预测", "queued": "10 步队列预测"}[key])
    axes[2].set_ylim(-.05, 1.1)
    axes[2].set_ylabel("夹爪命令：0 开，1 闭")
    axes[2].set_title("③ 夹爪：原始观测下能预测闭合与张开", loc="left", fontsize=12)
    axes[2].legend(loc="center left", fontsize=9)
    axes[2].set_xlabel("演示时间（秒）")
    for ax in axes:
        ax.set_xlim(0, t[-1]); ax.grid(alpha=.18)
    for when in metrics["target_gripper_transitions_s"]:
        axes[2].axvline(when, color="#888888", lw=.7, alpha=.6)
    fig.savefig(out/"prediction_overview.png"); fig.savefig(out/"prediction_overview.svg"); plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True, layout="constrained")
    for j, ax in enumerate(axes.flat):
        ax.plot(t, np.rad2deg(target[:, j]), color=colors["target"], lw=1, label="记录动作")
        ax.plot(t, np.rad2deg(fresh[:, j]), color=colors["fresh"], lw=.9, ls="--", label="每帧预测")
        ax.plot(t, np.rad2deg(queued[:, j]), color=colors["queued"], lw=.9, ls=":", label="10 步队列")
        ax.set_title(f"关节 {j+1}；每帧 MAE {metrics['fresh_first_action']['joint_mae_per_joint_deg'][j]:.2f}°")
        ax.set_ylabel("绝对目标角（度）"); ax.grid(alpha=.18); ax.set_xlim(0, t[-1])
    axes[0, 0].legend(fontsize=9)
    for ax in axes[-1]: ax.set_xlabel("演示时间（秒）")
    fig.suptitle("六个关节：记录目标与观测预测", fontsize=16)
    fig.savefig(out/"all_joint_predictions.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, when, title in zip(axes, metrics["target_gripper_transitions_s"], ["闭合阶段", "松开阶段"]):
        for key, pred, style in [("target", target, "-"), ("fresh", fresh, "--"), ("queued", queued, ":")]:
            ax.plot(t, np.clip(pred[:, 6], 0, 1), style, color=colors[key], lw=1.8,
                    label={"target": "记录动作", "fresh": "每帧预测", "queued": "10 步队列"}[key])
        ax.axhline(.5, color="#999999", ls="--", lw=.8)
        ax.axvline(when, color="#999999", ls=":", lw=.8)
        ax.set_xlim(when-.8, when+.8); ax.set_ylim(-.05, 1.05)
        ax.set_title(title); ax.set_xlabel("演示时间（秒）"); ax.grid(alpha=.18)
    axes[0].set_ylabel("夹爪命令"); axes[0].legend(fontsize=9)
    fig.suptitle("放大夹爪切换：横线 0.5 用于统计开闭", fontsize=14)
    fig.savefig(out/"gripper_transition_zoom.png"); plt.close(fig)
    shutil.copy2(Path(__file__), out/Path(__file__).name)
    manifest = json.loads((out/"manifest.json").read_text(encoding="utf-8"))
    manifest["files"][str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest["plot_command"] = "python tools/plot_training_fit.py " + str(out)
    (out/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved prediction overview, all joints, and gripper timing plots")


if __name__ == "__main__":
    main()
