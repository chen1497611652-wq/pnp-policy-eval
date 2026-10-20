"""调用现有评测循环，逐控制步记录状态及命令，供停滞/进展诊断。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import src.evaluate as ev


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path)
    parser.add_argument("--single-cup-demo", action="store_true",
                        help="Use single-cup XML and first episode obj_init, preserving the controller")
    args = parser.parse_args()
    ev.load_config(args.config)
    ev.require_layout()
    if ev.DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Configured GPU is unavailable")
    # 本工具一次只诊断一个 seed，日志与视频的相位关系写入说明文件。
    if len(ev.EVAL_SEEDS) != 1:
        raise ValueError("Trace one seed per run")
    seed = ev.EVAL_SEEDS[0]
    ev.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = ev.OUTPUT_DIR / f"trajectory_seed{seed}.jsonl"
    result_path = ev.OUTPUT_DIR / "results.jsonl"
    if trace_path.exists() or result_path.exists():
        raise FileExistsError("Trace/results already exist; choose a new output directory")
    reference = None
    env_factory = None
    demo_first_state = None
    if args.single_cup_demo:
        if args.reference_results:
            raise ValueError("Single-cup run is a different condition, not a prefix-reproduction check")
        import pandas as pd
        from src.single_cup_env import make_single_cup_env
        from PIL import Image
        source = sorted((ev.DATASET_ROOT / "data").rglob("*.parquet"))[0]
        data = pd.read_parquet(source, columns=["obj_init", "observation.state", "frame_index"])
        if int(data["frame_index"].iloc[0]) != 0:
            raise ValueError("Expected episode frame zero")
        demo_init = np.asarray(data["obj_init"].iloc[0]).copy()
        demo_first_state = np.asarray(data["observation.state"].iloc[0]).copy()

        def env_factory(xml_path, **kwargs):
            env = make_single_cup_env(xml_path, obj_init=demo_init, **kwargs)
            report = {**env.initialization_report, "source": str(source),
                      "demo_first_state": demo_first_state.tolist()}
            (ev.OUTPUT_ROOT / "initialization.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            image, _ = env.grab_image()
            Image.fromarray(image).save(ev.OUTPUT_ROOT / "singlecup_after_reset.png")
            return env
    if args.reference_results:
        rows = [json.loads(s) for s in args.reference_results.read_text(encoding="utf-8").splitlines() if s.strip()]
        reference = next(r for r in rows if r["seed"] == seed)
    policy_path = Path(ev.POLICY_PATH) if ev.POLICY_PATH else ev.find_pretrained_model(ev.MODEL_RUN_DIR)
    print(f"seed={seed}, device={ev.DEVICE}, steps={ev.MAX_ACTION_STEPS}, hz={ev.ENV_HZ}", flush=True)
    policy = ev.load_policy(ev.POLICY_TYPE, policy_path)
    n_rows = 0
    first_sim_time = None
    prefix_check = None
    with trace_path.open("x", encoding="utf-8") as trace_file:
        def record(row: dict) -> None:
            nonlocal n_rows, first_sim_time, prefix_check
            if first_sim_time is None:
                first_sim_time = row["sim_time_s"]
            row["observed_elapsed_s"] = row["sim_time_s"] - first_sim_time
            if n_rows == 0 and demo_first_state is not None:
                state = np.array(row["ee_pose"])
                diff = state - demo_first_state
                first = {"recorded_first_control": row, "demo_first_state": demo_first_state.tolist(),
                         "position_gap_mm": float(np.linalg.norm(diff[:3])*1000),
                         "max_rpy_component_gap_deg": float(np.rad2deg(np.abs(np.arctan2(np.sin(diff[3:]), np.cos(diff[3:])))).max())}
                (ev.OUTPUT_ROOT / "first_control_check.json").write_text(json.dumps(first, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"FIRST CONTROL vs demo: position gap={first['position_gap_mm']:.3f}mm, "
                      f"max RPY gap={first['max_rpy_component_gap_deg']:.3f}deg", flush=True)
            for name in ["state", "ee_pose", "joint_state", "held_target", "action_raw", "action_sent", "target_pos", "plate_pos"]:
                if not np.isfinite(np.asarray(row[name], dtype=float)).all():
                    raise ValueError(f"Nonfinite {name}, step={row['step']}")
            if len(row["action_raw"]) != 7 or len(row["state"]) != 6:
                raise ValueError("Unexpected state/action shape")
            if reference is not None and row["step"] + 1 == reference["action_steps"]:
                differences = {key: [reference[key], value] for key, value in row["metrics"].items()
                               if reference.get(key) != value}
                prefix_check = {"reference": str(args.reference_results), "step_count": row["step"] + 1,
                                "all_metrics_exactly_equal": not differences, "differences": differences}
                print("PREFIX CHECK " + json.dumps(prefix_check), flush=True)
            trace_file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            n_rows += 1
            if n_rows == 1 or n_rows % 200 == 0:
                trace_file.flush()
                print(f"step={n_rows}/{ev.MAX_ACTION_STEPS} observed_s={row['observed_elapsed_s']:.2f} "
                      f"tcp_z={row['ee_pose'][2]:.6f} gripper={row['action_sent'][6]:.4f} "
                      f"max_lift={row['metrics']['max_target_lift']:.8f}", flush=True)

        result = ev.run_one_seed(policy, seed, trace_callback=record, env_factory=env_factory)
    result_path.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    report = {"trace_rows": n_rows, "configured_steps": ev.MAX_ACTION_STEPS, "hz": ev.ENV_HZ,
              "nominal_budget_s": ev.MAX_ACTION_STEPS / ev.ENV_HZ,
              "phase": "Each row observes physics before its new action takes effect. held_target is the prior command. Final new command has no later physics step.",
              "prefix_check": prefix_check, "result": result}
    (ev.OUTPUT_DIR / "trace_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if prefix_check and not prefix_check["all_metrics_exactly_equal"]:
        print("WARNING: prefix differs; inspect before comparing horizons", flush=True)


if __name__ == "__main__":
    main()
