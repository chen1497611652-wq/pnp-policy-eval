"""单杯诊断：演示动作重放或同权重 ACT，对照状态轨迹与严格任务结果。"""
from __future__ import annotations

import argparse
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


class ReplayPolicy:
    """只取演示指令；末尾保持最后一条，便于检查稳定放置。"""
    def __init__(self, actions, device):
        self.actions = torch.as_tensor(actions, dtype=torch.float32, device=device)
        self.reset()

    def reset(self):
        self.index = 0

    def select_action(self, observation):
        index = min(self.index, len(self.actions)-1)
        self.index += 1
        return self.actions[index:index+1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["replay", "act"], required=True)
    parser.add_argument("--n-action-steps", type=int)
    parser.add_argument("--temporal-ensemble", type=float,
                        help="ACT temporal ensembling coefficient; recompute every step")
    args = parser.parse_args()
    ev.load_config(args.config)
    ev.require_layout()
    if ev.EVAL_SEEDS != [6]:
        raise ValueError("This diagnostic uses seed 6")
    if args.output.exists():
        raise FileExistsError("Choose a new diagnostic output directory")
    args.output.mkdir(parents=True)
    ev.OUTPUT_ROOT = args.output
    ev.OUTPUT_DIR = args.output / ("demo_replay_closed_loop" if args.mode == "replay" else "act_closed_loop")
    ev.OUTPUT_DIR.mkdir()
    import pandas as pd
    from src.single_cup_env import make_single_cup_env
    source = sorted((ev.DATASET_ROOT / "data").rglob("*.parquet"))[0]
    data = pd.read_parquet(source, columns=["action", "observation.state", "obj_init"])
    actions = np.stack(data["action"])
    states = np.stack(data["observation.state"])
    obj_init = np.asarray(data["obj_init"].iloc[0])
    print(f"MODE={args.mode} steps={ev.MAX_ACTION_STEPS} hz={ev.ENV_HZ}", flush=True)
    if args.mode == "replay":
        policy = ReplayPolicy(actions, ev.DEVICE)
        ev.POLICY_TYPE = "demo_replay"
    else:
        policy = ev.load_policy(ev.POLICY_TYPE, Path(ev.POLICY_PATH))
        if args.n_action_steps is not None:
            if not 1 <= args.n_action_steps <= policy.config.chunk_size:
                raise ValueError("n-action-steps outside chunk size")
            policy.config.n_action_steps = args.n_action_steps
        if args.temporal_ensemble is not None:
            if not np.isfinite(args.temporal_ensemble) or args.temporal_ensemble < 0:
                raise ValueError("temporal-ensemble must be finite and nonnegative")
            from lerobot.common.policies.act.modeling_act import ACTTemporalEnsembler
            policy.config.n_action_steps = 1
            policy.config.temporal_ensemble_coeff = args.temporal_ensemble
            policy.temporal_ensembler = ACTTemporalEnsembler(args.temporal_ensemble, policy.config.chunk_size)
        print(f"ACT n_action_steps={policy.config.n_action_steps}", flush=True)

    def factory(xml_path, **kwargs):
        env = make_single_cup_env(xml_path, obj_init=obj_init, **kwargs)
        (args.output/"initialization.json").write_text(json.dumps(env.initialization_report, indent=2), encoding="utf-8")
        return env

    n_rows = 0
    first_time = None
    with (ev.OUTPUT_DIR/"trajectory_seed6.jsonl").open("x", encoding="utf-8") as f:
        def record(row):
            nonlocal n_rows, first_time
            if first_time is None:
                first_time = row["sim_time_s"]
            row["observed_elapsed_s"] = row["sim_time_s"] - first_time
            i = row["step"]
            for key in ["ee_pose", "action_raw", "action_sent", "joint_state"]:
                assert np.isfinite(np.asarray(row[key])).all()
            if i < len(states):
                diff = np.asarray(row["ee_pose"]) - states[i]
                row["demo_position_gap_mm"] = float(np.linalg.norm(diff[:3])*1000)
                row["demo_rpy_component_gap_deg"] = float(np.rad2deg(np.abs(np.arctan2(np.sin(diff[3:]), np.cos(diff[3:])))).max())
            f.write(json.dumps(row, allow_nan=False) + "\n")
            n_rows += 1
            if n_rows == 1 or n_rows % 200 == 0:
                f.flush()
                print(f"step={n_rows} t={row['observed_elapsed_s']:.2f}s z={row['ee_pose'][2]:.5f} "
                      f"g={row['action_sent'][6]:.3f} lift={row['metrics']['max_target_lift']:.4f} "
                      f"demo_gap_mm={row.get('demo_position_gap_mm', float('nan')):.3f}", flush=True)
        result = ev.run_one_seed(policy, 6, trace_callback=record, env_factory=factory)
    (ev.OUTPUT_DIR/"results.jsonl").write_text(json.dumps(result, ensure_ascii=False)+"\n", encoding="utf-8")
    rows = [json.loads(s) for s in (ev.OUTPUT_DIR/"trajectory_seed6.jsonl").read_text(encoding="utf-8").splitlines()]
    gaps = [r["demo_position_gap_mm"] for r in rows if "demo_position_gap_mm" in r]
    report = {"mode":args.mode, "n_action_steps":policy.config.n_action_steps if args.mode == "act" else None,
              "temporal_ensemble_coeff":args.temporal_ensemble,
              "trajectory_rows":n_rows, "demo_position_gap_mm":{"median":float(np.median(gaps)), "max":float(np.max(gaps))},
              "result":result,
              "notes":["Replay uses recorded commands and is not model inference." if args.mode == "replay" else "ACT uses live images and measured states; no action labels are supplied.",
                       "Step ordering, dynamics, state input, and success thresholds follow original evaluator."]}
    (args.output/"summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    files = [Path(__file__), args.config, REPO/"src/evaluate.py", REPO/"src/single_cup_env.py", REPO/"src/success.py", source,
             Path(ev.POLICY_PATH)/"model.safetensors", Path(ev.POLICY_PATH)/"config.json"]
    manifest = {"command":" ".join(sys.argv), "files":{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (args.output/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for p in files[:5]:
        shutil.copy2(p,args.output/p.name)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
