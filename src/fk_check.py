"""FK 检验: 判断一份 LeRobot 数据集里 action 是命令还是实测关节角。

用法:
    python -m src.fk_check --parquet <episode.parquet> [--frames 0,500,...]
    --frames 不传时, 按行数自动取等分点。

只适用于 observation.state 是末端位姿(ee_pose)的数据集 —— 本工具把 action(关节角)
正解到末端位姿, 再拿去和 state 比。若数据集的 state 本身就是关节角(如
omy_pnp_language), 两边不是同一种量, 位置差会大得离谱, 此时判定无意义(会给出警告)。
"""


import argparse
from pathlib import Path
import numpy as np

def load_episode(parquet: Path) -> tuple[np.ndarray, np.ndarray]:
    """读一集数据 返回(action[N,7],states[N,6])。纯IO，不用MuJoCo"""
    import pandas as pd
    df =pd.read_parquet(parquet)
    actions =np.stack([np.asarray(a, dtype=np.float64) for a in df["action"]])
    states =np.stack([np.asarray(s, dtype=np.float64) for s in df["observation.state"]])
    return actions, states

def default_frames(n: int, k: int = 5) -> list[int]:
    """按行数自动取 k 个等分帧号(含首尾)。不写死帧号 ———— D8 就是写死 2320, 而数据集只有 2259 行, 直接 IndexError。"""
    if n == 0:
        return []
    if k <= 1:
        return [0]
    if n <= k:
        return list(range(n))
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def fk_pose(env, joint_q: np.ndarray) -> np.ndarray:
    env.env.set_qpos_joints(env.joint_names, joint_q[:6])
    return np.asarray(env.get_ee_pose(), dtype=np.float64)

def compare_pose(state: np.ndarray, ee: np.ndarray) -> tuple[float, float]:
    d = np.abs(np.asarray(state, dtype=np.float64)-np.asarray(ee, dtype=np.float64))
    return float(d[:3].max()), float(d[3:].max())

FLOAT32_NOISE_M = 1.2e-7      # float32 在 1 米量级的表示精度 ≈ 0.12 微米
MAX_PLAUSIBLE_M = 0.5         # 超过这个数就不像"命令 vs 实测"的位姿差了

def verdict(pos_errs: list[float]) -> None:
    worst = max(pos_errs)
    print()
    if worst > MAX_PLAUSIBLE_M:
        print(f"[??] 最大位置差 {worst:.3f} m —— 不像位姿差, 无法判定")
        print("   -> 确认这份数据集的 observation.state 是末端位姿(ee_pose)吗?")
    elif worst < FLOAT32_NOISE_M:
        print(f"[BAD] 最大位置差 {worst:.2e} m —— 落在 float32 噪声内")
        print("   -> action 记的是【实测关节角】, 数据集是坏的")
    else:
        print(f"[OK] 最大位置差 {worst*1e3:.3f} mm = 数值噪声的 {worst/FLOAT32_NOISE_M:.0f} 倍")
        print("   -> action 是【命令】, 标签语义正确")

def build_env(config: Path):
    """只有这里需要 MuJoCo。project_root 从 configs/default.yaml 读, 不写死。"""
    import sys, yaml
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    project_root = Path(cfg["paths"]["project_root"]).expanduser()
    sys.path.insert(0, str(project_root))
    from mujoco_env.y_env2 import SimpleEnv2
    xml_path = project_root / cfg["env"]["xml"]
    return SimpleEnv2(str(xml_path), action_type=cfg["env"]["action_type"],
                      state_type=cfg["env"]["state_type"], seed=0)

def frames_arg(text: str) -> list[int]:
    """把 --frames 的 "0,500,1000" 解析成 [0, 500, 1000]。"""
    return [int(part) for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FK 检验: 判断数据集的 action 是命令还是实测关节角"
    )
    parser.add_argument("--parquet", type=Path, required=True,
                        help="一集 episode 的 parquet 路径(其 observation.state 须为末端位姿 ee_pose)")
    parser.add_argument("--frames", type=frames_arg, default=None,
                        help="逗号分隔的帧号, 如 0,500,1000; 不传则按行数自动取等分点")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[1] / "configs" / "default.yaml",
                        help="YAML 配置(提供 project_root 与 env)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    actions, states = load_episode(args.parquet)
    n = len(actions)
    frames = args.frames or default_frames(n)
    if not frames:
        raise SystemExit("没有可抽样的帧: 数据集是空的")
    out_of_range = [i for i in frames if not 0 <= i < n]
    if out_of_range:
        raise SystemExit(
            f"帧号越界: {out_of_range}  (数据集只有 {n} 行, 合法范围 0..{n-1})"
        )
    print(f"数据集 : {args.parquet}")
    print(f"帧数   : {n}    抽样帧: {frames}")
    print()
    env = build_env(args.config)
    errs = []
    for i in frames:
        p, r = compare_pose(states[i], fk_pose(env, actions[i]))
        errs.append(p)
        print(f"帧 {i:>5}:  位置差 {p:.8f} m  |  姿态差 {r:.8f} rad")
    verdict(errs)


if __name__ == "__main__":
    main()

