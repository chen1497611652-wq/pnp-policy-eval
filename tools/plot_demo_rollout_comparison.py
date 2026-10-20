"""为原演示与 seed 6 双杯闭环对照作图；只读已生成的数据，不运行模型。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


DEMO = "#0072B2"
LIVE = "#D55E00"
PURPLE = "#7B4D9F"
INK = "#263342"


def configure_style() -> None:
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 170,
        "svg.fonttype": "none",
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
    })


def save_figure(fig: plt.Figure, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def early_plot(data: dict[str, np.ndarray], output: Path) -> None:
    t = data["t"]
    selected = t <= 3.0 + 1e-8
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    fig.subplots_adjust(top=.86, bottom=.18, hspace=.44, left=.11, right=.97)
    fig.suptitle("从开场看差异：两次运行并非从同一个画面、同一个姿态出发", y=.97,
                 fontsize=17, weight="bold")
    fig.text(.11, .91, "各自第一条记录记为 0 秒；下图只描述时间上的差异，不据此判断谁导致了谁。",
             fontsize=11, color="#596675")

    specs = [
        ("position_gap_mm", "① 末端位置：两次记录中的手相隔多远", "三维位置距离（毫米）", DEMO),
        ("orientation_component_gap_deg", "② 末端朝向：三个姿态分量中，哪项相差最大", "最大姿态分量差（度）", PURPLE),
    ]
    for ax, (key, title, ylabel, color) in zip(axes, specs):
        values = data[key]
        ax.plot(t[selected], values[selected], color=color, lw=2)
        ax.scatter(t[0], values[0], s=34, color=color, zorder=4)
        unit = "毫米" if key == "position_gap_mm" else "度"
        ax.annotate(f"起点已差 {values[0]:.2f} {unit}", xy=(t[0], values[0]),
                    xytext=(11, -18 if key == "position_gap_mm" else 11),
                    textcoords="offset points", fontsize=10, color=color)
        ax.set_title(title, loc="left", fontsize=12, weight="bold", pad=10)
        ax.set_ylabel(ylabel)
        ax.margins(y=.22)

    axes[2].plot(t[selected], data["action_gap_vs_demo_policy_deg"][selected],
                 color=LIVE, lw=2, label="双杯输出 vs 原演示输入下的 ACT 输出")
    axes[2].plot(t[selected], data["action_gap_vs_demo_target_deg"][selected],
                 color="#4B5563", lw=1.6, ls="--", label="双杯输出 vs 原演示记录的目标动作")
    axes[2].set_title("③ 预测动作：六个关节目标平均相差多少", loc="left", fontsize=12,
                      weight="bold", pad=10)
    axes[2].set_ylabel("六关节平均绝对差（度）")
    axes[2].legend(fontsize=9, loc="best", frameon=False)
    axes[2].margins(y=.2)
    axes[2].set_xlabel("从各自第一条记录开始的时间（秒）")
    for ax in axes:
        for when in np.arange(.5, 3.01, .5):
            ax.axvline(when, color="#ACB4BD", lw=.8, ls=":", alpha=.8, zorder=0)
        ax.set_xlim(0, min(3.0, float(t[-1])))
        ax.grid(axis="y", alpha=.17)
    fig.text(.11, .065, "细竖虚线：每 0.5 秒读取新观测，预测下一组 10 个动作（20 Hz）。\n"
             "姿态差是 roll / pitch / yaw 的逐项环绕差最大值，不是完整三维旋转角；动作比较不含夹爪。",
             fontsize=9, color="#596675", linespacing=1.6)
    save_figure(fig, output, "early_divergence")


def full_plot(data: dict[str, np.ndarray], output: Path) -> None:
    t, demo, live = data["t"], data["demo_states"], data["live_states"]
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    fig.subplots_adjust(top=.88, bottom=.145, hspace=.36, left=.11, right=.97)
    fig.suptitle("拉长时间看：原演示与双杯闭环走向哪里", y=.97, fontsize=17, weight="bold")
    fig.text(.11, .923, "按各自第一条记录对齐时间；只展示共同记录时段，并未把两次初始条件设成相同。",
             fontsize=10.5, color="#596675")
    for j, (ax, name) in enumerate(zip(axes[:3], ["x 方向", "y 方向", "z 高度"])):
        ax.plot(t, demo[:, j] * 100, color=DEMO, lw=1.4, label="原演示")
        ax.plot(t, live[:, j] * 100, color=LIVE, lw=1.3, label="seed 6 双杯闭环")
        ax.set_ylabel(f"{name}（厘米）")
        ax.grid(alpha=.17)
        ax.margins(y=.13)
    axes[0].legend(loc="best", ncol=2, fontsize=10, frameon=False)
    axes[3].plot(t, data["orientation_component_gap_deg"], color=PURPLE, lw=1.3)
    axes[3].set_ylabel("最大姿态分量差（度）")
    axes[3].set_xlabel("从各自第一条记录开始的时间（秒）")
    axes[3].grid(alpha=.17)
    for ax in axes:
        ax.set_xlim(0, float(t[-1]))
    fig.text(.11, .045, "前三幅是两次运行的末端坐标；第四幅是姿态分量差，不与位置共用单位。\n"
             "时间对齐方便找差异，并不意味着两次运行必须在同一秒完成同一阶段。",
             fontsize=9, color="#596675", linespacing=1.6)
    save_figure(fig, output, "trajectory_comparison")


def image_pairs(output: Path) -> bool:
    steps = [0, 10, 20, 40]
    paths = [(output / f"model_demo_{s:04d}.png", output / f"model_live_{s:04d}.png") for s in steps]
    if not all(path.exists() for pair in paths for path in pair):
        return False
    fig, axes = plt.subplots(2, 4, figsize=(13, 8))
    fig.subplots_adjust(left=.02, right=.98, top=.86, bottom=.12, hspace=.26, wspace=.07)
    fig.suptitle("给模型看的画面：开场已经不同", y=.97, fontsize=18, weight="bold")
    fig.text(.02, .91, "每一对：左边原演示，右边双杯闭环。时间从各自第一条记录开始计算。",
             fontsize=11, color="#596675")
    for i, (step, pair) in enumerate(zip(steps, paths)):
        row, col = divmod(i, 2)
        for offset, (path, title, color) in enumerate(zip(pair, ["原演示", "双杯闭环（重建）"], [DEMO, LIVE])):
            ax = axes[row, 2 * col + offset]
            ax.imshow(plt.imread(path), interpolation="nearest")
            ax.set_title(f"{step / 20:.2f} 秒  |  {title}", fontsize=11, color=color, pad=8)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(1.5)
    fig.text(.02, .055, "右侧按闭环记录的物理状态重建相机画面，并应用评测的图像处理；不是再次运行控制策略。\n"
             "可观察物体数量、摆放和机械臂姿态；画面差异本身不能证明是哪一项导致了动作差异。",
             fontsize=9.5, color="#596675", linespacing=1.6)
    save_figure(fig, output, "early_observation_pairs")
    return True


def local_sensitivity(output: Path) -> bool:
    path = output / "summary.json"
    if not path.exists():
        return False
    rows = json.loads(path.read_text(encoding="utf-8")).get("local_swaps", [])
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.subplots_adjust(top=.8, bottom=.24, left=.11, right=.97)
    fig.suptitle("只替换一部分输入：动作对哪一部分更敏感", y=.97, fontsize=17, weight="bold")
    fig.text(.11, .88, "基线：原演示的图片 + 状态。每次重置动作队列，只比较下一步关节目标。",
             fontsize=10.5, color="#596675")
    x = np.arange(len(rows))
    specs = [
        ("only_image_changed_first_action_mae_deg", "只换成双杯图片", DEMO),
        ("only_state_changed_first_action_mae_deg", "只换成闭环状态", PURPLE),
        ("both_changed_first_action_mae_deg", "图片、状态都换", LIVE),
    ]
    for i, (key, label, color) in enumerate(specs):
        values = [row[key] for row in rows]
        bars = ax.bar(x + (i - 1) * .24, values, width=.22, color=color, label=label)
        ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=4, fontsize=9)
    ax.set_xticks(x, [f"{row['time_s']:g} 秒" for row in rows])
    ax.set_ylabel("六关节预测相对基线的平均绝对差（度）")
    ax.set_xlabel("从各自第一条记录开始的时间")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.17)
    ax.margins(y=.2)
    ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=10)
    fig.text(.11, .065, "这是同一时刻的局部输入替换：图片和状态可以来自不同运行，属于人为组合。\n"
             "状态包含位置 xyz 和姿态 roll / pitch / yaw；三种结果不能简单相加，也不能直接确定整段失败的原因。",
             fontsize=9.5, color="#596675", linespacing=1.6)
    save_figure(fig, output, "local_input_sensitivity")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="compare_demo_rollout.py 的输出目录")
    args = parser.parse_args()
    with np.load(args.output / "comparison.npz") as arrays:
        data = {key: arrays[key] for key in arrays.files}
    required = {
        "t", "demo_states", "live_states", "position_gap_mm", "orientation_component_gap_deg",
        "action_gap_vs_demo_policy_deg", "action_gap_vs_demo_target_deg",
    }
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Missing comparison arrays: {sorted(missing)}")
    n = len(data["t"])
    if n < 2 or not np.all(np.diff(data["t"]) > 0):
        raise ValueError("Expected at least two samples with increasing time")
    for key in required:
        if len(data[key]) != n or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid comparison array: {key}")
    if data["demo_states"].shape != (n, 6) or data["live_states"].shape != (n, 6):
        raise ValueError("Expected six state components per sample")
    configure_style()
    early_plot(data, args.output)
    full_plot(data, args.output)
    paired = image_pairs(args.output)
    sensitivity = local_sensitivity(args.output)
    print(f"Saved early_divergence, trajectory_comparison; image pairs: {paired}; sensitivity: {sensitivity}")


if __name__ == "__main__":
    main()
