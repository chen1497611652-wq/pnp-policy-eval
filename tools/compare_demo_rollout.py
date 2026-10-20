"""对齐原演示与已保存的闭环，重放旧指令补拍画面，并检查局部输入敏感性。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import src.evaluate as ev


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--offline", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ev.load_config(args.config)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out/"summary.json").exists():
        raise FileExistsError("Completed comparison already exists")
    rows = [json.loads(s) for s in args.trace.read_text(encoding="utf-8").splitlines() if s.strip()]
    archive = np.load(args.offline/"predictions.npz")
    n = min(len(rows), len(archive["times"]))
    t = np.array([r["observed_elapsed_s"] for r in rows[:n]])
    np.testing.assert_allclose(t, archive["times"][:n], rtol=0, atol=1e-5)
    demo_state = archive["states"][:n]
    demo_target = archive["target"][:n]
    demo_pred = archive["queued"][:n]
    live_state = np.array([r["ee_pose"] for r in rows[:n]])
    live_action = np.array([r["action_sent"] for r in rows[:n]])
    xyz_gap = np.linalg.norm(live_state[:, :3]-demo_state[:, :3], axis=1)*1000
    rpy_gap = live_state[:, 3:]-demo_state[:, 3:]
    orientation_gap = np.rad2deg(np.abs(np.arctan2(np.sin(rpy_gap), np.cos(rpy_gap)))).max(1)
    action_gap = np.rad2deg(np.abs(live_action[:, :6]-demo_pred[:, :6])).mean(1)
    target_gap = np.rad2deg(np.abs(live_action[:, :6]-demo_target[:, :6])).mean(1)
    np.savez_compressed(out/"comparison.npz", t=t, demo_states=demo_state, live_states=live_state,
                        demo_target=demo_target, demo_policy_queued=demo_pred, live_action=live_action,
                        position_gap_mm=xyz_gap, orientation_component_gap_deg=orientation_gap,
                        action_gap_vs_demo_policy_deg=action_gap, action_gap_vs_demo_target_deg=target_gap)
    with (out/"aligned_comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "time_s", "position_gap_mm", "max_rpy_component_gap_deg", "action_gap_vs_demo_policy_deg", "action_gap_vs_demo_target_deg"])
        for i in range(n):
            writer.writerow([i,t[i],xyz_gap[i],orientation_gap[i],action_gap[i],target_gap[i]])

    import pandas as pd
    paths = sorted((ev.DATASET_ROOT/"data").rglob("*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    capture = [0, 10, 20, 40, 100]
    demo_images = {}
    for i in capture:
        cell = df["observation.image"].iloc[i]
        if not cell.get("bytes"):
            raise ValueError("Expected embedded original image bytes")
        rgb = np.asarray(Image.open(BytesIO(cell["bytes"])).convert("RGB"))
        demo_images[i] = ev.image_tensor(rgb)
        Image.fromarray((demo_images[i].permute(1,2,0).numpy()*255).round().astype(np.uint8)).save(out/f"model_demo_{i:04d}.png")

    # 只重放之前发出的命令，逐点核对物理轨迹；本段不调用模型。
    from mujoco_env.y_env2 import SimpleEnv2
    torch.manual_seed(6); np.random.seed(6)
    env = SimpleEnv2(str(ev.PROJECT_ROOT/ev.ENV_XML), action_type=ev.ACTION_TYPE,
                     state_type=ev.STATE_TYPE, seed=6)
    live_images = {}
    max_state_replay_error = 0.
    max_joint_replay_error = 0.
    first_raw_matches = False
    try:
        env.set_instruction(ev.TASK_TEXT)
        i = 0
        while i <= max(capture):
            env.step_env()
            if not env.env.loop_every(HZ=ev.ENV_HZ):
                continue
            ee = np.asarray(env.get_ee_pose())
            joint = np.asarray(env.get_joint_state())
            state_error = float(np.abs(ee-np.array(rows[i]["ee_pose"])).max())
            joint_error = float(np.abs(joint-np.array(rows[i]["joint_state"])).max())
            max_state_replay_error = max(max_state_replay_error, state_error)
            max_joint_replay_error = max(max_joint_replay_error, joint_error)
            np.testing.assert_array_equal(ee, np.array(rows[i]["ee_pose"]))
            np.testing.assert_array_equal(joint, np.array(rows[i]["joint_state"]))
            assert abs(float(env.env.data.time)-rows[i]["sim_time_s"]) < 1e-10
            if i in capture:
                rgb, _ = env.grab_image()
                live_images[i] = ev.image_tensor(rgb)
                Image.fromarray(rgb).save(out/f"replay_raw_{i:04d}.png")
                Image.fromarray((live_images[i].permute(1,2,0).numpy()*255).round().astype(np.uint8)).save(out/f"model_live_{i:04d}.png")
                if i == 0:
                    old_rgb = np.asarray(Image.open(args.preflight/"agentview_seed6.png").convert("RGB"))
                    first_raw_matches = bool(np.array_equal(rgb, old_rgb))
                    assert first_raw_matches, "Replayed first camera differs from archived raw camera"
                print(f"replay verified through step {i}, t={t[i]:.2f}s", flush=True)
            env.step(np.asarray(rows[i]["action_sent"], dtype=np.float32))
            i += 1
    finally:
        env.env.close_viewer()

    # 2×2诊断：同一时刻分别只换图片、只换状态、同时换两项。
    policy = ev.load_policy(ev.POLICY_TYPE, Path(ev.POLICY_PATH))
    assert set(policy.config.input_features) == {"observation.image", "observation.state"}
    images, states = [], []
    for i in capture:
        images.extend([demo_images[i],live_images[i],demo_images[i],live_images[i]])
        states.extend([demo_state[i],demo_state[i],live_state[i],live_state[i]])
    obs = {"observation.image": torch.stack(images).to(ev.DEVICE),
           "observation.state": torch.as_tensor(np.stack(states), dtype=torch.float32, device=ev.DEVICE)}
    assert set(obs) == {"observation.image", "observation.state"}
    policy.reset()
    with torch.inference_mode():
        pred = torch.stack([policy.select_action(obs) for _ in range(10)], dim=1).cpu().numpy()
    pred = pred.reshape(len(capture), 4, 10, 7)
    assert np.isfinite(pred).all()
    swaps = []
    for slot, i in enumerate(capture):
        p = pred[slot]
        recorded = np.array([r["action_raw"] for r in rows[i:i+10]])
        original = archive["all_chunks"][i]
        demo_reproduction_error = float(np.abs(p[0]-original).max())
        live_reproduction_error = float(np.abs(p[3]-recorded).max())
        np.testing.assert_allclose(p[0], original, rtol=0, atol=2e-5)
        np.testing.assert_allclose(p[3], recorded, rtol=0, atol=2e-5)
        effect = lambda x: float(np.rad2deg(np.abs(p[x,0,:6]-p[0,0,:6])).mean())
        interactions = p[3,0,:6]-p[1,0,:6]-p[2,0,:6]+p[0,0,:6]
        swaps.append({"step": i, "time_s": float(t[i]),
                      "only_image_changed_first_action_mae_deg": effect(1),
                      "only_state_changed_first_action_mae_deg": effect(2),
                      "both_changed_first_action_mae_deg": effect(3),
                      "interaction_mae_deg": float(np.rad2deg(np.abs(interactions)).mean()),
                      "image_mean_abs_pixel_difference_0_1": float(torch.abs(demo_images[i]-live_images[i]).mean()),
                      "original_prediction_reproduction_max_abs": demo_reproduction_error,
                      "live_prediction_reproduction_max_abs": live_reproduction_error,
                      "first_actions": p[:,0,:].tolist()})
    np.savez_compressed(out/"local_swaps.npz", steps=capture, predictions=pred,
                        image_keys=np.array(["demo", "live", "demo", "live"]),
                        state_keys=np.array(["demo", "demo", "live", "live"]))
    points = []
    for i in [0, 1, 9, 10, 11, 20, 40, 100, 200, 300]:
        points.append({"step":i, "time_s":float(t[i]), "position_gap_mm":float(xyz_gap[i]),
                       "max_rpy_component_gap_deg":float(orientation_gap[i]),
                       "action_gap_vs_demo_policy_deg":float(action_gap[i]),
                       "action_gap_vs_demo_target_deg":float(target_gap[i])})
    first_crossings = {}
    for threshold in [5, 10, 20, 50]:
        idx=np.flatnonzero(xyz_gap>threshold)
        first_crossings[str(threshold)+"mm"] = float(t[idx[0]]) if len(idx) else None
    summary = {"aligned_frames":n, "aligned_span_s":float(t[-1]),
               "max_clock_difference_s":float(np.abs(t-archive["times"][:n]).max()),
               "alignment": "Both first observations set to zero; equal elapsed time does not imply equal initial conditions or task progress.",
               "initial_demo_obj_init":df["obj_init"].iloc[0].tolist(),
               "initial_live_objects":json.loads((args.preflight/"preflight.json").read_text(encoding="utf-8"))["object_positions"],
               "initial_position_gap_mm":float(xyz_gap[0]),
               "initial_max_rpy_component_gap_deg":float(orientation_gap[0]),
               "initial_offline_prediction_vs_demo_target_mae_deg":float(np.rad2deg(np.abs(demo_pred[0,:6]-demo_target[0,:6])).mean()),
               "timepoints":points, "first_position_crossings_descriptive_only":first_crossings,
               "local_swaps":swaps,
               "replay_check":{"steps_checked":101, "max_ee_component_abs_difference":max_state_replay_error,
                               "max_joint_abs_difference":max_joint_replay_error,
                               "first_raw_image_matches_archived_preflight":first_raw_matches},
               "phase":"Observations precede the new action; command at .50s can affect the .55s observation, not the observation in its own row.",
               "scope":"Image/state swaps probe local input sensitivity, not the cause of the full failure; scene differences are intentional generalization conditions."}
    (out/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    source_paths = [args.config,args.trace,args.offline/"predictions.npz",args.preflight/"agentview_seed6.png",Path(ev.POLICY_PATH)/"model.safetensors",Path(ev.POLICY_PATH)/"config.json",Path(__file__),*paths]
    manifest={"date_local":"2026-10-20", "torch":torch.__version__,"command":" ".join(sys.argv),
              "files":{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}}
    (out/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(Path(__file__),out/Path(__file__).name)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
