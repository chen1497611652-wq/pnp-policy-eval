"""汇总单杯诊断结果、真实闭环曲线和视频关键帧，不重新运行仿真。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    trials = []
    result = json.loads((args.baseline / "results.jsonl").read_text(encoding="utf-8").strip())
    trials.append(("original", result, args.baseline / "trajectory_seed6.jsonl"))
    for path in sorted(args.root.glob("*/summary.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        traces = list(path.parent.glob("*_closed_loop/trajectory_seed6.jsonl"))
        if len(traces) != 1:
            raise ValueError(f"Expected one trace: {path}")
        trials.append((path.parent.name, report["result"], traces[0]))
    compact = []
    for name, result, path in trials:
        rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
        closed = [r["control_time_s"] for r in rows if r["action_sent"][6] > .5]
        lifted = [r["control_time_s"] for r in rows if r["metrics"]["max_target_lift"] >= .03]
        compact.append({"name": name, **result, "trace": str(path),
                        "first_command_closed_s": min(closed) if closed else None,
                        "first_lift_3cm_s": min(lifted) if lifted else None})
    (args.output / "experiments.json").write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = {"original", "replay", "act_yaw_mean", "act_finetuned_yaw_noise"}
    colors = {"original": "#d55e00", "replay": "#009e73", "act_yaw_mean": "#cc79a7", "act_finetuned_yaw_noise": "#0072b2"}
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for name, result, path in trials:
        if name not in selected:
            continue
        rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
        t = [r["control_time_s"] for r in rows]
        data = ([r["ee_pose"][2] for r in rows],
                [r["metrics"]["max_target_lift"] * 100 for r in rows],
                [r["action_sent"][6] for r in rows])
        for ax, values in zip(axes, data):
            ax.plot(t, values, label=name, color=colors[name], lw=1.2, alpha=.85)
    for ax, label in zip(axes, ["TCP height (m)", "Max mug lift so far (cm)", "Grip command (0 open, 1 closed)"]):
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].axhline(3, color="gray", ls="--", lw=.8)
    axes[2].set_xlabel("Control time since first observation (s)")
    fig.suptitle("Single-cup seed 6: same physics and strict success thresholds")
    fig.savefig(args.output / "comparison.png", dpi=170)
    fig.savefig(args.output / "comparison.svg")
    plt.close(fig)

    import cv2
    from PIL import Image, ImageDraw
    times = [0., 30., 85., 119.]
    visual = [trial for trial in trials if trial[0] in selected]
    tile_w, tile_h = 320, 275
    sheet = Image.new("RGB", (len(times) * tile_w, len(visual) * tile_h), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (name, result, _) in enumerate(visual):
        capture = cv2.VideoCapture(result["video"])
        if not capture.isOpened():
            raise ValueError(result["video"])
        for col, time_s in enumerate(times):
            capture.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000)
            ok, frame = capture.read()
            if not ok:
                # Successful episodes can end before 110 s: show the final available frame.
                capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, capture.get(cv2.CAP_PROP_FRAME_COUNT) - 1))
                ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode frame: {name} {time_s}")
            actual_s = (capture.get(cv2.CAP_PROP_POS_FRAMES) - 1) / capture.get(cv2.CAP_PROP_FPS)
            rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((320, 243))
            x, y = col * tile_w, row * tile_h
            sheet.paste(rgb, (x, y + 32))
            draw.text((x + 4, y + 3), f"{name}\nvideo t={actual_s:.1f}s", fill="black")
        capture.release()
    sheet.save(args.output / "keyframes.jpg", quality=92)
    print(json.dumps([{k: r[k] for k in ["name", "physical_success", "legacy_success", "max_target_lift", "first_command_closed_s", "first_lift_3cm_s"]} for r in compact], indent=2))


if __name__ == "__main__":
    main()
