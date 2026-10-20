"""使用原始演示观测检查部署模式 ACT 动作预测；推理不接收动作标签。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import src.evaluate as ev


def score(pred, target):
    if len(target) == 0:
        return None
    error = pred[:, :6] - target[:, :6]
    grip = np.clip(pred[:, 6], 0., 1.)
    wanted_closed = target[:, 6] >= .5
    predicted_closed = grip >= .5
    return {
        "n": len(target),
        "joint_mae_rad": float(np.abs(error).mean()),
        "joint_mae_deg": float(np.rad2deg(np.abs(error).mean())),
        "joint_mae_per_joint_deg": np.rad2deg(np.abs(error).mean(0)).tolist(),
        "joint_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(error**2)))),
        "joint_abs_error_p95_deg": float(np.rad2deg(np.quantile(np.abs(error), .95))),
        "gripper_raw_mae": float(np.abs(pred[:, 6]-target[:, 6]).mean()),
        "gripper_clipped_mae": float(np.abs(grip-target[:, 6]).mean()),
        "gripper_accuracy": float(np.mean(predicted_closed == wanted_closed)),
        "gripper_closed_recall": float(predicted_closed[wanted_closed].mean()) if wanted_closed.any() else None,
        "gripper_open_recall": float((~predicted_closed[~wanted_closed]).mean()) if (~wanted_closed).any() else None,
        "gripper_closed_frames": int(wanted_closed.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    ev.load_config(args.config)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out/"predictions.npz").exists() or (out/"metrics.json").exists():
        raise FileExistsError("Predictions already exist; choose a new output directory")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    import pandas as pd
    from safetensors import safe_open
    dataset = LeRobotDataset(ev.DATASET_REPO_ID, root=str(ev.DATASET_ROOT), image_transforms=None)
    source_paths = sorted((ev.DATASET_ROOT/"data").rglob("*.parquet"))
    frame = pd.concat([pd.read_parquet(p) for p in source_paths], ignore_index=True)
    if frame["episode_index"].nunique() != 1:
        raise ValueError("This diagnostic currently supports one demonstration episode")
    actions = np.stack(frame["action"]).astype(np.float32)
    states = np.stack(frame["observation.state"]).astype(np.float32)
    times = frame["timestamp"].to_numpy(dtype=float)
    assert len(dataset) == len(frame)
    assert np.array_equal(frame["frame_index"].to_numpy(), np.arange(len(frame)))
    print(f"dataset={ev.DATASET_ROOT}, frames={len(dataset)}, time={times[-1]:.2f}s", flush=True)
    assert torch.cuda.is_available() or not ev.DEVICE.startswith("cuda")
    policy = ev.load_policy(ev.POLICY_TYPE, Path(ev.POLICY_PATH))
    assert not policy.training and not policy.model.training
    keys = list(policy.config.input_features)
    assert set(keys) == {"observation.image", "observation.state"}
    horizon = policy.config.n_action_steps
    assert horizon == policy.config.chunk_size == 10
    assert policy.config.action_delta_indices == list(range(horizon))
    assert policy.config.temporal_ensemble_coeff is None
    print(f"Only inference keys: {keys}; fresh chunks of {horizon} actions", flush=True)
    stat_check = {}
    with safe_open(str(Path(ev.POLICY_PATH)/"model.safetensors"), framework="np") as ckpt:
        for feature, values, prefix in [("state", states, "normalize_inputs.buffer_observation_state"),
                                        ("action", actions, "unnormalize_outputs.buffer_action")]:
            for stat, calc in [("mean", values.mean(0)), ("std", values.std(0))]:
                saved = ckpt.get_tensor(prefix+"."+stat)
                loaded = policy.state_dict()[prefix+"."+stat].detach().cpu().numpy()
                stat_check[feature+"_"+stat] = {"dataset_checkpoint_max_abs_diff": float(np.max(np.abs(calc-saved))),
                                               "loaded_checkpoint_exact": bool(np.array_equal(loaded, saved))}
                assert np.array_equal(loaded, saved)

    all_predictions = np.empty((len(dataset), horizon, 7), dtype=np.float32)
    torch.manual_seed(6)
    with torch.inference_mode():
        for start in range(0, len(dataset), args.batch_size):
            stop = min(start+args.batch_size, len(dataset))
            items = [dataset[i] for i in range(start, stop)]
            # 白名单取观测；不能把含 action 的整条样本传给 ACT 的 VAE。
            obs = {k: torch.stack([s[k] for s in items]).to(ev.DEVICE) for k in keys}
            np.testing.assert_array_equal(obs["observation.state"].cpu().numpy(), states[start:stop])
            assert torch.isfinite(obs["observation.image"]).all()
            assert 0 <= obs["observation.image"].min() <= obs["observation.image"].max() <= 1
            policy.reset()
            # 第一次调用预测整块；后九次只从队列取动作，不换题、不再次推理。
            chunk = torch.stack([policy.select_action(obs) for _ in range(horizon)], dim=1)
            assert tuple(chunk.shape) == (stop-start, horizon, 7)
            assert torch.isfinite(chunk).all()
            all_predictions[start:stop] = chunk.cpu().numpy()
            if start == 0 or stop == len(dataset) or stop % 256 == 0:
                print(f"predicted {stop}/{len(dataset)} original observations", flush=True)

    fresh = all_predictions[:, 0]
    # 每十帧使用一次真实观测，按原部署队列取十个动作；绝不把预测送回状态。
    queued = np.empty_like(actions)
    for start in range(0, len(dataset), horizon):
        count = min(horizon, len(dataset)-start)
        queued[start:start+count] = all_predictions[start, :count]
    full_starts = np.arange(0, len(dataset)-horizon+1, horizon)
    full_targets = np.stack([actions[i:i+horizon] for i in full_starts])
    speed = np.r_[0., np.linalg.norm(np.diff(states[:, :3], axis=0), axis=1)*dataset.fps]
    moving = speed > .005
    mean_baseline = np.broadcast_to(actions.mean(0), actions.shape)
    median_baseline = np.broadcast_to(np.median(actions, axis=0), actions.shape)
    shift = int(10*dataset.fps)
    assert shift < len(dataset)
    metrics = {
        "frames": len(dataset), "fps": dataset.fps, "duration_last_timestamp_s": float(times[-1]),
        "inference_keys": keys, "mode": "eval, no action labels, no image augmentation",
        "chunk_size": horizon, "normalization_check": stat_check,
        "fresh_first_action": score(fresh, actions),
        "queued_ten_actions": score(queued, actions),
        "complete_nonoverlapping_chunks": {"chunks": len(full_starts), "score": score(all_predictions[full_starts].reshape(-1, 7), full_targets.reshape(-1, 7))},
        "moving_frames_over_5mm_s": {"count": int(moving.sum()), "fresh": score(fresh[moving], actions[moving]), "queued": score(queued[moving], actions[moving])},
        "baselines": {"constant_mean": score(mean_baseline, actions), "constant_median": score(median_baseline, actions),
                      "previous_target_on_frames_1_to_end": score(actions[:-1], actions[1:])},
        "negative_control_10s_label_shift": {"correct_alignment_same_subset": score(fresh[:-shift], actions[:-shift]),
                                            "wrong_alignment": score(fresh[:-shift], actions[shift:])},
        "closed_frame_range": [int(i) for i in np.flatnonzero(actions[:, 6] >= .5)[[0, -1]]],
        "target_gripper_transitions_s": times[1:][np.diff((actions[:, 6]>=.5).astype(int)) != 0].tolist(),
        "fresh_gripper_transitions_s": times[1:][np.diff((np.clip(fresh[:, 6], 0, 1)>=.5).astype(int)) != 0].tolist(),
        "queued_gripper_transitions_s": times[1:][np.diff((np.clip(queued[:, 6], 0, 1)>=.5).astype(int)) != 0].tolist(),
        "scope": "Teacher-forced original observations, not a closed-loop success test; no causal scene attribution.",
    }
    np.savez_compressed(out/"predictions.npz", times=times, states=states, target=actions,
                        fresh=fresh, queued=queued, all_chunks=all_predictions, speed_m_s=speed)
    (out/"metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out/"per_frame.csv").open("w", newline="", encoding="utf-8-sig") as f:
        names = ["frame", "time_s", "moving_over_5mm_s", "fresh_joint_mae_deg", "queued_joint_mae_deg", "target_gripper", "fresh_gripper_raw", "queued_gripper_raw"]
        writer = csv.writer(f); writer.writerow(names)
        for i in range(len(dataset)):
            writer.writerow([i, times[i], int(moving[i]), np.rad2deg(np.abs(fresh[i, :6]-actions[i, :6]).mean()),
                             np.rad2deg(np.abs(queued[i, :6]-actions[i, :6]).mean()), actions[i, 6], fresh[i, 6], queued[i, 6]])
    sample_indices = np.unique(np.array([0, 500, 1000, 1552, 1800, 2173]).clip(0, len(dataset)-1))
    from PIL import Image, ImageDraw
    sheet = Image.new("RGB", (900, 640), "white"); draw = ImageDraw.Draw(sheet)
    examples = []
    for slot, i in enumerate(sample_indices):
        item = dataset[int(i)]
        image = Image.fromarray((item["observation.image"].permute(1, 2, 0).numpy()*255).round().astype(np.uint8))
        image.save(out/f"training_frame_{i:04d}.png")
        x=(slot%3)*300; y=(slot//3)*320
        sheet.paste(image, (x+22, y+28))
        draw.text((x+10, y+5), f"frame {i} / t={times[i]:.2f}s", fill="black")
        draw.text((x+10, y+290), f"grip true={actions[i,6]:.2f} pred={fresh[i,6]:.2f}", fill="black")
        examples.append({"frame": int(i), "time_s": float(times[i]), "state": states[i].tolist(), "target": actions[i].tolist(), "prediction_fresh": fresh[i].tolist()})
    sheet.save(out/"training_examples.jpg", quality=94)
    (out/"examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
    paths = [*source_paths, Path(ev.POLICY_PATH)/"model.safetensors", Path(ev.POLICY_PATH)/"config.json", Path(__file__), args.config,
             ev.DATASET_ROOT/"meta/info.json", ev.DATASET_ROOT/"meta/stats.json"]
    manifest = {"date_local": "2026-10-20", "torch": torch.__version__, "device": ev.DEVICE,
                "gpu": torch.cuda.get_device_name(0) if ev.DEVICE.startswith("cuda") else None,
                "batch_size": args.batch_size, "command": " ".join(sys.argv),
                "files": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}}
    (out/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(Path(__file__), out/Path(__file__).name)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
