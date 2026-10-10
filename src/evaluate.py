"""基于固定 seed 的 MuJoCo 闭环策略评测。

改编自 datawhalechina/every-embodied 专题 04-AMD-ROCm策略复刻专题
的 code/run_closed_loop.py (CC BY 4.0, Copyright (c) Datawhale and contributors)。
本版本改动: 配置外置到 YAML; 移除模块级 argparse 与 os.chdir;
video_io 内联; state_type/action_type 参数化。

用法:
    python -m src.evaluate --config configs/default.yaml

ROCm 下 PyTorch 仍使用 cuda 设备名(LeRobot/ROCm 的兼容约定)；
默认不开 MuJoCo 窗口，需要时把 runtime.render 设为 true。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path



TOPIC_ROOT = Path(__file__).resolve().parents[1]


# 模块顶层: 只留占位, 不留逻辑
PROJECT_ROOT = DATASET_ROOT = OUTPUT_ROOT = OUTPUT_DIR = MODEL_ROOT = None
MODEL_RUN_DIR = None
POLICY_TYPE = TASK_TEXT = DATASET_REPO_ID = POLICY_PATH = DEVICE = None
EVAL_SEEDS = MAX_ACTION_STEPS = VIDEO_STRIDE = VIDEO_FPS = None
RENDER = STATE_TYPE = ACTION_TYPE = STATE_SLICE = ENV_HZ = ENV_XML = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="固定 seed 的 MuJoCo 闭环评测")
    parser.add_argument("--config", type=Path, required=True,
                        help="YAML 配置文件路径")
    return parser.parse_args()


def load_config(path: Path) -> None:
    """读 YAML, 填充模块级配置变量。"""
    global PROJECT_ROOT, DATASET_ROOT, OUTPUT_ROOT, OUTPUT_DIR, MODEL_ROOT
    global MODEL_RUN_DIR
    global POLICY_TYPE, TASK_TEXT, DATASET_REPO_ID, POLICY_PATH, DEVICE
    global EVAL_SEEDS, MAX_ACTION_STEPS, VIDEO_STRIDE, VIDEO_FPS
    global RENDER, STATE_TYPE, ACTION_TYPE, STATE_SLICE, ENV_HZ, ENV_XML

    import yaml          # 你环境里已有 pyyaml 6.0.3, 我确认过
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    PROJECT_ROOT = Path(cfg["paths"]["project_root"]).expanduser()
    OUTPUT_ROOT  = Path(cfg["paths"]["output_root"]).expanduser()
    MODEL_ROOT   = PROJECT_ROOT / "ckpt"

    DATASET_REPO_ID = cfg["dataset"]["repo_id"]
    DATASET_ROOT    = Path(cfg["dataset"]["root"]).expanduser()

    POLICY_TYPE = cfg["policy"]["type"]
    POLICY_PATH = str(cfg["policy"].get("path") or "").strip()
    DEVICE      = cfg["policy"]["device"]

    ENV_XML     = cfg["env"]["xml"]
    ACTION_TYPE = cfg["env"]["action_type"]
    STATE_TYPE  = cfg["env"]["state_type"]
    STATE_SLICE = int(cfg["env"]["state_slice"])
    ENV_HZ      = int(cfg["env"]["hz"])
    TASK_TEXT   = cfg["env"]["task_text"]

    EVAL_SEEDS       = [int(s) for s in cfg["eval"]["seeds"]]
    MAX_ACTION_STEPS = max(1, int(cfg["eval"]["max_action_steps"]))

    VIDEO_STRIDE = max(1, int(cfg["video"]["stride"]))
    VIDEO_FPS    = max(1, int(cfg["video"]["fps"]))

    RENDER = bool(cfg["runtime"]["render"])

    # 派生量: 与原脚本的 OUTPUT_DIR = OUTPUT_ROOT / f"{POLICY_TYPE}_closed_loop" 同义
    OUTPUT_DIR = OUTPUT_ROOT / f"{POLICY_TYPE}_closed_loop"
    # 原 --model-run-dir / $env:MODEL_RUN_DIR 都没给时的 fallback 分支
    MODEL_RUN_DIR = MODEL_ROOT / f"{POLICY_TYPE}_rocm_full"

    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(f"找不到 project_root: {PROJECT_ROOT}")

    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from src.success import read_metrics, update_metrics

def require_layout() -> None:
    required = [
        PROJECT_ROOT / ENV_XML,
        PROJECT_ROOT / "mujoco_env" / "y_env2.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    info_path = DATASET_ROOT / "meta" / "info.json"
    if not info_path.exists():
        missing.append(str(info_path))
    if missing:
        raise FileNotFoundError(
            "运行闭环评估前缺少以下文件：\n- " + "\n- ".join(missing)
        )


def find_pretrained_model(run_dir: Path) -> Path:
    """同时接受直接的 pretrained_model 目录和训练输出目录。"""
    run_dir = Path(run_dir).expanduser()
    if not run_dir.exists():
        raise FileNotFoundError(f"模型目录不存在：{run_dir}")

    direct_markers = (
        "config.json",
        "train_config.json",
        "model.safetensors",
        "pytorch_model.bin",
    )
    if run_dir.is_dir() and any((run_dir / marker).exists() for marker in direct_markers):
        return run_dir

    last = run_dir / "checkpoints" / "last" / "pretrained_model"
    if last.exists():
        return last
    candidates = sorted((run_dir / "checkpoints").glob("*/pretrained_model"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"没有在 {run_dir} 下找到 pretrained_model。")


def image_tensor(array: np.ndarray, size: tuple[int, int] = (256, 256)) -> torch.Tensor:
    from PIL import Image

    image = Image.fromarray(array).convert("RGB").resize(size)
    value = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(value).permute(2, 0, 1).contiguous()


def load_policy(policy_type: str, policy_path: Path):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    if policy_type == "act":
        from lerobot.common.policies.act.modeling_act import ACTPolicy as policy_class
    elif policy_type == "smolvla":
        from lerobot.common.policies.smolvla.modeling_smolvla import (
            SmolVLAPolicy as policy_class,
        )
    elif policy_type == "pi0":
        from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy as policy_class
    else:
        raise ValueError("POLICY_TYPE 只能是 act、smolvla 或 pi0。")

    metadata = LeRobotDatasetMetadata(
        DATASET_REPO_ID,
        root=str(DATASET_ROOT),
    )
    policy = policy_class.from_pretrained(
        str(policy_path),
        dataset_stats=metadata.stats,
    )
    policy.to(DEVICE)
    policy.eval()
    return policy

##接受前几步积累的统计量，返回一个包含全部指标的字典和更新后的 stable_place_steps
def strict_snapshot(env, initial_plate_pos, max_target_lift,
                    max_lifted_run, stable_place_steps):
    """薄适配层：签名与两个调用点保持不变，实现搬到 src/success.py。"""
    return update_metrics(
        read_metrics(env, initial_plate_pos),
        max_target_lift=max_target_lift,
        max_lifted_run=max_lifted_run,
        stable_place_steps=stable_place_steps,
    )



def run_one_seed(policy, seed: int) -> dict:
    from src.video_io import write_video
    from mujoco_env.y_env2 import SimpleEnv2

    torch.manual_seed(seed)
    np.random.seed(seed)

    xml_path = PROJECT_ROOT / ENV_XML
    
    env = SimpleEnv2(
        str(xml_path),
        action_type=ACTION_TYPE,
        state_type=STATE_TYPE,
        seed=seed,
    )
    env.set_instruction(TASK_TEXT)
    if hasattr(policy, "reset"):
        policy.reset()

    initial_plate_pos = np.asarray(
        env.env.get_p_body("body_obj_plate_11"), dtype=np.float64
    ).copy()
    initial_target_z = float(env.env.get_p_body(env.obj_target)[2])
    max_target_lift = 0.0
    lifted_run = 0
    max_lifted_run = 0
    stable_place_steps = 0
    frames: list[np.ndarray] = []
    final = None
    action_step = 0
    started = time.perf_counter()

    try:
        # 不渲染时允许无窗口运行；RENDER=1 才要求 viewer 保持存活。
        while action_step < MAX_ACTION_STEPS and (
            not RENDER or env.env.is_viewer_alive()
        ):
            env.step_env()
            if not env.env.loop_every(HZ=ENV_HZ):
                continue

            state = env.get_joint_state()[:STATE_SLICE]
            agent_image, wrist_image = env.grab_image()
            observation = {
                "observation.state": torch.as_tensor(
                    state, dtype=torch.float32, device=DEVICE
                ).unsqueeze(0),
                "observation.image": image_tensor(agent_image)
                .to(DEVICE)
                .unsqueeze(0),
                "observation.wrist_image": image_tensor(wrist_image)
                .to(DEVICE)
                .unsqueeze(0),
                "task": [TASK_TEXT],
            }
            with torch.inference_mode():
                action = (
                    policy.select_action(observation)[0, :7]
                    .detach()
                    .cpu()
                    .numpy()
                )
            action = action.astype(np.float32)
            action[6] = np.clip(action[6], 0.0, 1.0)
            env.step(action)

            target_z = float(env.env.get_p_body(env.obj_target)[2])
            lift = target_z - initial_target_z
            max_target_lift = max(max_target_lift, lift)
            lifted_run = lifted_run + 1 if lift >= 0.03 else 0
            max_lifted_run = max(max_lifted_run, lifted_run)
            final, stable_place_steps = strict_snapshot(
                env,
                initial_plate_pos,
                max_target_lift,
                max_lifted_run,
                stable_place_steps,
            )

            if action_step % VIDEO_STRIDE == 0:
                frames.append(np.asarray(agent_image).copy())
            if RENDER:
                env.render()
            action_step += 1
            if final["physical_success"]:
                break

        if final is None:
            final, _ = strict_snapshot(
                env,
                initial_plate_pos,
                max_target_lift,
                max_lifted_run,
                stable_place_steps,
            )

        video_path = OUTPUT_DIR / f"{POLICY_TYPE}_seed{seed}.mp4"
        video_saved = False
        if frames:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            try:
                write_video(video_path, frames, VIDEO_FPS)
                video_saved = True
            except Exception as exc:
                print(f"视频保存失败，仍保留 JSONL 结果：{exc!r}", flush=True)

        return {
            "policy_type": POLICY_TYPE,
            "seed": int(seed),
            "instruction": TASK_TEXT,
            "action_steps": int(action_step),
            "elapsed_s": round(time.perf_counter() - started, 3),
            "video": str(video_path) if video_saved else None,
            **final,
        }
    finally:
        close_viewer = getattr(env.env, "close_viewer", None)
        if close_viewer is not None:
            close_viewer()


def main() -> None:
    load_config(parse_args().config) 
    require_layout()
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "当前 DEVICE=cuda，但 torch.cuda.is_available() 为 False；"
            "请确认 ROCm/PyTorch 环境，或显式设置 DEVICE=cpu 做接口排查。"
        )

    policy_path = (
        Path(POLICY_PATH).expanduser()
        if POLICY_PATH
        else find_pretrained_model(MODEL_RUN_DIR)
    )
    print("TOPIC_ROOT =", TOPIC_ROOT)
    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("DATASET_ROOT =", DATASET_ROOT)
    print("MODEL_RUN_DIR =", MODEL_RUN_DIR)
    print("POLICY_PATH =", policy_path)
    print("POLICY_TYPE =", POLICY_TYPE)
    print("DEVICE =", DEVICE, "RENDER =", RENDER)
    print("EVAL_SEEDS =", EVAL_SEEDS)

    policy = load_policy(POLICY_TYPE, policy_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT_DIR / "results.jsonl"
    results = []
    for seed in EVAL_SEEDS:
        result = run_one_seed(policy, seed)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        result_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results),
            encoding="utf-8",
        )
    physical = sum(bool(row.get("physical_success")) for row in results)
    legacy = sum(bool(row.get("legacy_success")) for row in results)
    print(f"完成：legacy_success={legacy}/{len(results)}")
    print(f"完成：physical_success={physical}/{len(results)}")
    print("结果文件：", result_path)


if __name__ == "__main__":
    main()
