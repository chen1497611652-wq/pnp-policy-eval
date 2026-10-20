# DAY18：ACT 输入接入与受限闭环检查

整理日期：2026-10-20。**先核对训练与部署的数据含义，再用模型结果作判断。**

本次已修复按配置读取状态的路径，并检查 ACT 的加载、动作接口、原始演示预测与单 seed 闭环。
当前测试为 33 项通过；闭环未成功不等于输入读取修复未完成。

## 核心改动与验收

- `src/evaluate.py`：`read_state()` 按 `joint_angle` / `ee_pose` 调用对应 getter；非法类型及切片长度报错，循环实际调用该函数。
- `tests/test_state.py`：两个分支、关节夹爪切片、非法类型、零/负/超长切片。
- 用户已执行负控：故意接错 ee_pose getter 后测试失败，随后恢复。未保存那次终端原始输出，此项依据会话报告。
- `configs/act_y.yaml`：ACT 权重、demo_data、六维 ee_pose 输入、七维 joint_angle 输出。
- `src/single_cup_env.py`：在本仓库内适配单杯初始化，沿用 SimpleEnv2 控制和成功判定；不修改上游。
- `tests/test_single_cup_env.py`：7 项替身检查，覆盖单杯布局、初始化顺序与沉降、无蓝杯访问、继承接口、非法任务/数据。
- 可选轨迹回调及环境工厂支持额外诊断；默认评测未开启额外追踪。
- 默认 SmolVLA 配置和 `src/success.py` 未修改。**本次未运行新的 SmolVLA 真实闭环回归。**

## 版本库内的小型证据

| 条件 | 动作数 / 名义预算 | 严格成功 | 最大杯子抬升 | 原始结果 |
|---|---|---|---|---|
| 双杯，seed 6 | 300 / 15 秒 | False | 约 0.0013 毫米 | [JSONL](../results/act_y_d18_double_15s_seed6.jsonl) |
| 双杯，seed 6 | 2400 / 120 秒 | False | 约 0.0013 毫米 | [JSONL](../results/act_y_d18_double_120s_seed6.jsonl) |
| 演示物体起点的单杯，seed 6 | 2400 / 120 秒 | False | 约 0.536 毫米 | [JSONL](../results/act_y_d18_singlecup_demo_120s_seed6.jsonl) |

这些是不同条件的单次试跑，不作为同条件重复实验或总体成功率估计。
单杯、双杯后 105 秒均表现为末端局部小幅波动、夹爪保持张开、任务停滞。
单杯恢复了杯盘的记录起点，未恢复完整历史 qpos/qvel/control；首帧末端仍差约 1.54 毫米、最大姿态分量约 0.30°。

- [训练样本预测指标](../results/act_y_d18_training_fit_metrics.json)：2259 帧，推理只输入图片和状态；逐帧首动作平均六关节误差约 0.561°，10 步队列约 0.760°；运动片段队列约 1.564°；个别单关节误差约 30°。
- 阈值 0.5 下，夹爪开闭准确率约 99.87%，闭合召回率 100%。它能在原演示观测下预测闭爪，不能由此保证闭环成功。
- 数据有较多静止/平滑阶段，上一帧真实目标参考误差仅 0.109°，低平均误差不是充分能力证明。
- 故意错位 10 秒，2059 帧同子集关节 MAE 从约 0.561° 增到 15.650°，评分能发现时间错位。
- [归档来源及 SHA256](../results/act_y_d18_evidence_manifest.json)：三份结果及离线指标均逐字节复制；原始记录里的视频路径仍指向本机，不代表版本库包含视频。

## 本机完整产物

根目录：`D:/lerobot-mujoco-tutorial/outputs/`。大文件留在本机，本页仅索引。

| 目录 | 内容 |
|---|---|
| `d18_act_y_seed6` | 真实图片/状态预检查，模型归一化检查，15 秒试跑 |
| `d18_act_y_seed6_120s` | 双杯长试跑、2400 行状态/动作轨迹、视频、曲线与诊断说明 |
| `d18_act_y_training_check` | 原演示预测、全部动作块、指标、原图及曲线 |
| `d18_act_y_seed6_divergence` | 早期分叉时间线、原始图片重放、局部图片/状态替换 |
| `d18_act_y_singlecup_demo_seed6_120s` | 单杯初始化记录、首帧偏差、2400 行轨迹、结果和视频 |

局部诊断：0.5 秒第一次重规划时动作与原演示输入下的预测分叉约 5.10°。
该点只换图片约 0.18°，只换六维状态约 5.12°。这只说明局部输入敏感性，不能锁定整段失败原因；状态尚未拆分位置/姿态。

## 运行方式

在仓库根目录，使用已安装依赖的 Python 环境：

```powershell
& "D:/miniconda3-1/envs/py310/python.exe" -m pytest -q
& "D:/miniconda3-1/envs/py310/python.exe" -m src.evaluate --config configs/act_y.yaml
& "D:/miniconda3-1/envs/py310/python.exe" tools/trace_rollout.py --config configs/act_y_120s.yaml
& "D:/miniconda3-1/envs/py310/python.exe" tools/trace_rollout.py --config configs/act_y_singlecup_demo_120s.yaml --single-cup-demo
```

已有输出是证据。需要重跑时先复制配置并设置新输出目录；追踪工具会拒绝覆盖已有轨迹/结果。
本次收尾不要求重跑这些仿真命令。

## 结论边界与遗留

| 可以说 | 不能据此说 |
|---|---|
| 当前状态读取分支、加载和推理链路已验证 | ACT 已全面可评测，所有环境均可用 |
| 本机、该权重、seed 6 的单杯和双杯条件均停滞 | ACT 没用，或者根因一定是蓝杯 |
| 原演示观测下能近似复现动作趋势 | 原题闭环一定能成功，或模型已证明使用图像 |
| 单杯记录物体起点已恢复 | 历史仿真完整状态已逐位恢复 |

**已完成**：状态修复、测试、加载检查、受限试跑和追加诊断。
**暂缓**：D25 两权重正式多 seed A/B，待另立条件与验收；不在 D18 继续追加实验。
**证据不足**：闭环停滞根因、单杯精确历史状态、新 SmolVLA 回归。

后续按原计划做 D19 复盘。诊断线索可列入待办，不作为 D18 未完成的理由。
