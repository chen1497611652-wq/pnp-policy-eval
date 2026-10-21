"""检查单段演示的重复观测，以及原模型对闭环状态各分量的局部敏感性。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.evaluate as ev


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ev.load_config(args.config)
    if args.output.exists():
        raise FileExistsError(args.output)
    import pandas as pd
    from safetensors import safe_open
    source = sorted((ev.DATASET_ROOT / "data").rglob("*.parquet"))
    frame = pd.concat([pd.read_parquet(p) for p in source], ignore_index=True)
    if frame["episode_index"].nunique() != 1:
        raise ValueError("This diagnostic requires one episode")
    states = np.stack(frame["observation.state"]).astype(np.float32)
    actions = np.stack(frame["action"])
    times = np.asarray(frame["timestamp"], dtype=float)
    groups = {}
    for i, (state, image) in enumerate(zip(states, frame["observation.image"])):
        key = (state.tobytes(), hashlib.sha256(image["bytes"]).digest())
        groups.setdefault(key, []).append(i)
    duplicates = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        duplicates.append({"count": len(indices), "first": indices[0], "last": indices[-1],
                           "time_first": float(times[indices[0]]), "time_last": float(times[indices[-1]]),
                           "max_target_range_deg": float(np.rad2deg(np.ptp(actions[indices, :6], axis=0)).max()),
                           "grip_range": float(np.ptp(actions[indices, 6]))})
    speed = np.linalg.norm(np.diff(states[:, :3], axis=0), axis=1) / np.diff(times)
    rows = [json.loads(s) for s in args.trace.read_text(encoding="utf-8").splitlines() if s.strip()]
    live = np.asarray([r["ee_pose"] for r in rows], dtype=np.float32)
    checkpoint = Path(ev.POLICY_PATH) / "model.safetensors"
    with safe_open(str(checkpoint), framework="np") as weights:
        mean = weights.get_tensor("normalize_inputs.buffer_observation_state.mean")
        std = weights.get_tensor("normalize_inputs.buffer_observation_state.std")
    steps = [0, 10, 20, 40, 100]
    variants = {"demo": [], "live_xyz": [0, 1, 2], "live_rpy": [3, 4, 5],
                "live_roll": [3], "live_pitch": [4], "live_yaw": [5], "live_all": list(range(6))}
    images, samples = [], []
    for i in steps:
        rgb = np.asarray(Image.open(BytesIO(frame["observation.image"].iloc[i]["bytes"])).convert("RGB"))
        for axes in variants.values():
            sample = states[i].copy()
            sample[axes] = live[i, axes]
            samples.append(sample)
            images.append(ev.image_tensor(rgb))
    policy = ev.load_policy(ev.POLICY_TYPE, Path(ev.POLICY_PATH))
    policy.reset()
    with torch.inference_mode():
        predictions = policy.select_action({"observation.image": torch.stack(images).to(ev.DEVICE),
                                            "observation.state": torch.as_tensor(np.stack(samples), device=ev.DEVICE)})
    predictions = predictions.cpu().numpy().reshape(len(steps), len(variants), 7)
    assert np.isfinite(predictions).all()
    probes = []
    for slot, i in enumerate(steps):
        gap = live[i] - states[i]
        probes.append({"step": i, "time_s": rows[i].get("observed_elapsed_s", rows[i]["control_time_s"]),
                       "live_minus_demo": gap.tolist(), "gap_in_training_std": (gap / (std + 1e-8)).tolist(),
                       "first_action_change_mae_deg": {name: float(np.rad2deg(np.abs(predictions[slot, j, :6] - predictions[slot, 0, :6])).mean())
                                                       for j, name in enumerate(variants)}})
    report = {"frames": len(frame), "unique_exact_image_state_groups": len(groups),
              "stationary_transitions_below_0_5_mm_s": int((speed < .0005).sum()),
              "transition_count": len(speed), "duplicate_groups": duplicates,
              "contradictory_groups_above_0_05_deg": sum(g["max_target_range_deg"] > .05 for g in duplicates),
              "state_order": ["x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad"],
              "state_mean": mean.tolist(), "state_std": std.tolist(), "probes": probes,
              "scope": "Local swaps use original demo images and measured live state components, never action labels. Sensitivity is not proof of full rollout causality.",
              "files": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [*source, args.trace, args.config, checkpoint, Path(__file__)]}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["frames", "unique_exact_image_state_groups", "contradictory_groups_above_0_05_deg", "state_std", "probes"]}, indent=2))


if __name__ == "__main__":
    main()
