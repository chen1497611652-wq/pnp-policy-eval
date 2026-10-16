# pnp-policy-eval
基于MuJoCo的仿真，用严格物理判定+固定seed，对ACT/SmolVLA/Pi0做可复现的策略评测。
现有评测脚本判定条件宽松且结果不可复现，同一模型跑两次结论不同。
## 致谢

`src/evaluate.py` 与 `src/video_io.py` 改编自
[datawhalechina/every-embodied](https://github.com/datawhalechina/every-embodied)
「04-AMD-ROCm策略复刻专题」的 `code/run_closed_loop.py` 和 `code/video_io.py`。
原文以 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 发布，
Copyright (c) Datawhale and contributors。

本项目对其的改动：配置外置到 `configs/default.yaml`；移除模块级 `argparse` 与 `os.chdir`；
`video_io` 内联；`state_type` / `action_type` / 状态切片长度参数化。

## 发现

- [`docs/finding-01-action-semantics.md`](docs/finding-01-action-semantics.md)
  —— 教程采集脚本把 `action` 记成实测关节角而非命令目标，
  导致策略学到的只是静态正运动学映射。含可复现的 FK 检验工具
  （`src/fk_check.py`）。

- [`docs/finding-02-nondeterminism.md`](docs/finding-02-nondeterminism.md)
  —— 教程评测脚本从未给模型播种（`seed` 只固定了杯子摆在哪，
  管不到 flow matching 每步现抽的噪声），同一 seed 跑两次
  步数与结局都可能不同。含可复现的逐位比对工具
  （`src/nondeterminism_check.py`）。
