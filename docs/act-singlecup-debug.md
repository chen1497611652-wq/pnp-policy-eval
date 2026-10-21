# ACT 单杯失败：第一轮诊断与修改尝试

2026-10-20，分支 `codex/act-singlecup-debug`。**尚未修好单杯闭环；已复现失败，找出局部朝向敏感性，并完成两次微调及多种推理对照。** 所有失败结果都保留，成功标准没有降低。

**先把两种“没通过”分开。**

1. 原 ACT：机械臂局部移动后停滞，夹爪一直没有发出大于 0.5 的闭合命令，没有抓起杯子。
2. 演示命令重放：约 77.6 秒闭爪、82.4 秒杯子首次抬高 3 cm，最大抬高 18.6 cm，搬到盘子上；旧判定成功，但杯子最后倾斜，严格判定失败。

因此，当前环境和控制接口能够执行这段演示的抓取、搬运。不能把 ACT 的停滞简单归因于环境完全不能抓杯，也不能把重放等同于 ACT 学会。重放在当前环境中的末态，也不能代替历史演示当时的物理末态。

**这一轮实际跑了什么。**

全部使用单红杯、记录的杯盘起点、seed 6、20 Hz、最多 2400 条命令（名义 120 秒）。沿用原控制器、真实相机和严格判定。原始数据及原权重没有覆盖。原基线来自此前归档，其余为本次新运行，每个条件一次，不是成功率实验。

| 条件 | 杯子最大抬升 | 发出闭爪命令 >0.5 | 旧成功 / 严格成功 |
|---|---:|---|---|
| 原 ACT 基线 | 0.536 mm | 否 | 否 / 否 |
| 演示命令重放，不调用模型 | 185.953 mm | 是 | 是 / 否 |
| 原 ACT，每步重新预测，`n_action_steps=1` | 0.536 mm | 否 | 否 / 否 |
| 原 ACT，每步预测并做 temporal ensemble，系数 0.01 | 0.536 mm | 否 | 否 / 否 |
| 原权重继续训练 300 步，重复观测只采末次 | 0.536 mm | 否 | 否 / 否 |
| 原 ACT，给模型的 yaw 临时替换为训练均值 | 4.979 mm | 否 | 否 / 否 |
| 原权重继续训练 1000 步，训练输入 yaw 加噪声 | 0.536 mm | 否 | 否 / 否 |

抬升不足 3 cm 不算通过抬升门槛；小幅抖动、碰杯也不算抓取。yaw 均值替换只作诊断：它使末端下探到约 0.847 m，仍然停滞，不能作为已验证的修复启用。

重放的末态 `upright_cos=0.5756`，低于门槛 0.7；图像也能看到杯子倾斜。末态其他门槛满足，但直立条件不满足，所以没有形成稳定放置候选。重放末端轨迹与记录的 xyz 中位差约 0.000011 mm，最大差 9.75 mm；位置相近不表示所有接触、杯子旋转或历史初态一致。

小型证据：[实验结果](../results/act_singlecup_debug_20261020_experiments.json)、[原模型输入诊断](../results/act_singlecup_debug_20261020_inputs.json)、[候选输入诊断](../results/act_singlecup_debug_20261020_inputs_yaw_noise.json)、[归档哈希](../results/act_singlecup_debug_20261020_manifest.json)。

完整轨迹、模型、视频及图在本机 `D:/lerobot-mujoco-tutorial/outputs/act_singlecup_debug_20261020/`，不放入 Git。重点看 `report/keyframes.jpg` 和 `report/comparison.png`，视频中 85 秒可见重放已经提杯，119 秒可见盘中倾斜。

**证据最明确的线索：模型对很小的朝向偏差过度敏感。**

这里的 yaw 是末端朝向角。训练中它的标准差只有约 0.048°。归一化的计算是：

```text
归一化数值 = (测量值 - 训练均值) / 训练标准差
```

标准差太小时，很小的物理偏差也会变成明显的输入偏差。它不是“归一化公式写错了”，而是模型面对的实际变化超过了训练里常见的变化范围。

第 0.5 秒的同一观测点，保留**原演示图片**，只替换指定状态分量；比较新预测和原演示观测下的预测，六关节首动作平均绝对变化为：

| 只替换什么 | 原模型 | yaw 扰动训练后的候选 |
|---|---:|---:|
| xyz 位置 | 0.182° | 0.121° |
| roll | 0.023° | 0.040° |
| pitch | 0.054° | 0.034° |
| yaw | **5.157°** | **0.311°** |
| 全部六维状态 | 5.145° | 0.211° |

此处实际 yaw 只相差约 0.126°，约为训练标准差的 2.63 倍。**这证明这个局部预测对 yaw 敏感，不能证明 yaw 是整段停滞的唯一原因。** 后续原轨迹里 pitch 的偏差也会明显影响预测。固定 yaw 的闭环仍失败，进一步说明不能只处理这一项就宣布成功。

候选采用原模型、原采样帧、1000 个优化步、batch 16、AdamW lr 1e-5；只在训练输入 yaw 上添加标准差 0.005 rad（约 0.286°）的高斯扰动，动作标签不变。评测恢复真实输入，未做 yaw 替换。归一化缓冲值经逐项检查完全不变。

候选的该局部敏感性比原模型低，但任务依然失败；未做相同步数、无扰动的继续训练对照，因此不能把所有变化单独归功于加噪声，更不能宣称泛化能力已经提升。两个候选都是训练集上的探索，没有独立验证集。

**第二条线索：一段演示里有大量等待，观测有时不能区分“继续等”和“开始动”。**

`demo_data` 只有一集 2259 帧。2258 个相邻帧间隔中，1573 个（约 69.7%）位置速度小于 0.5 mm/s；这是位置静止口径，不等于所有关节完全不动。

按 float32 六维状态字节及原 PNG 字节完全相同分组，共 1372 种观测，40 组观测重复。其中 34 组存在六关节目标范围超过 0.05° 的不同标签。例如帧 60–93，3.0–4.65 秒，34 条输入完全相同，某个关节目标范围却相差约 1.91°。

可以把它想成：给模型看同一张照片、同一个位置，有的样本教它等待，有的教它开始动。模型没有人的意图或时间输入，就可能偏向更常见的等待。这是**数据歧义的实测证据和机制线索**，不是已经完成因果验证的唯一根因。

300 步候选仅让相同观测保留末次作为采样起点，仍取原始时间轴 `action[i:i+10]`，没有把删帧后的动作重新拼接，也没有修改源数据。结果仍停滞。因此“去掉重复就一定能成功”已不成立，不能把本次尝试包装成修复。

此前 [finding-01](finding-01-action-semantics.md) 已区分坏数据 `demo_data_example` 和重采后的 `demo_data`。本轮使用后者，不能因为这次同样悬停，就直接认定又是“action 记成当前关节角”的旧 bug。

**用容易理解的话串起来。**

离线预测像照着原题作答：每次都给模型演示者当时看到的图片和状态。闭环则要求它自己动一下，再根据自己产生的新画面继续做。第一步有一点偏差，后面的输入就可能不再是它熟悉的原题；如果它又对某个角度特别敏感，偏差就会扩大。大量等待样本还可能让它最终停在某个位置。

所以要分别检查三层：环境能否执行动作、模型能否持续产生有进展的动作、放下后是否满足直立和稳定标准。本轮第一层的重放对照已能抓取和搬运，第二层仍未解决，第三层在重放中也发现了独立问题。

**新增工具和验证。**

- `tools/debug_singlecup.py`：演示命令重放、每步重规划、temporal ensemble、显式 yaw 均值探针；保留真实状态，额外记录替换后给模型的状态。
- `tools/analyze_singlecup_inputs.py`：重复输入/不同标签检查、训练尺度、逐项输入替换实验。
- `tools/finetune_singlecup.py`：有预算的独立候选微调，显式采样方式及六维状态噪声；不覆盖源数据及权重。
- `tools/summarize_singlecup_debug.py`：汇总结果，输出曲线与实际视频关键帧。
- `tests/test_singlecup_diagnostics.py`：非连续重复采样、输入长度不匹配、探针不改原始观测、重放末尾保持及重置。

当前测试 **37 项通过**。这说明软件检查通过，不代表机器人任务通过。`src/evaluate.py`、`src/success.py` 和默认配置未在本轮追加修改。

本机证据另有一个来源说明：yaw 扰动微调启动后补了输入合法性检查和报告文字，训练算法未变；该次结束时的默认脚本快照包含后补检查。`finetune_yaw_noise/source_provenance.json` 与 `executed_source_reconstructed.py` 单独记录了实际执行版本的重建来源，避免把结束时磁盘文件误当成启动时源文件。后续工具已改为启动时保存脚本快照。

**下一步优先处理什么。**

先建立一条在当前严格标准下确实能直立放稳的单杯演示，并重放验证；否则即使模型完美模仿这次重放，严格任务仍可能失败。然后围绕同一单杯任务采集若干起点略有变化、动作连贯、包含偏离后纠正的演示，保留正确的命令标签与原始时间轴，再比较原模型与新训练模型。可以先采 5–10 条作小规模试验，这只是工作预算，不保证这个数量足以成功。

继续训练时，把“减少等待”“状态扰动”“继续训练步数”分别做对照；先看能否稳定接近和闭爪，再看抬升与严格放置。当前证据不足以宣布唯一根因，也不足以把任何候选替换为默认模型。

**复现命令示例**（每次换一个不存在的输出目录）：

```powershell
$py = 'D:/miniconda3-1/envs/py310/python.exe'
$out = 'D:/lerobot-mujoco-tutorial/outputs/act_singlecup_debug_repeat'
& $py -m pytest -q
& $py tools/debug_singlecup.py --config configs/act_y_singlecup_demo_120s.yaml --output "$out/replay" --mode replay
& $py tools/debug_singlecup.py --config configs/act_y_singlecup_demo_120s.yaml --output "$out/replan1" --mode act --n-action-steps 1
& $py tools/debug_singlecup.py --config configs/act_y_singlecup_demo_120s.yaml --output "$out/ensemble" --mode act --temporal-ensemble 0.01
& $py tools/debug_singlecup.py --config configs/act_y_singlecup_demo_120s.yaml --output "$out/yaw_probe" --mode act --yaw-mean
& $py tools/finetune_singlecup.py --config configs/act_y_singlecup_demo_120s.yaml --output "$out/finetune_last" --steps 300 --sampling last_duplicate
& $py tools/debug_singlecup.py --config "$out/finetune_last/candidate.yaml" --output "$out/eval_last" --mode act
& $py tools/finetune_singlecup.py --config configs/act_y_singlecup_demo_120s.yaml --output "$out/finetune_noise" --steps 1000 --sampling all --state-noise-std 0 0 0 0 0 0.005
& $py tools/debug_singlecup.py --config "$out/finetune_noise/candidate.yaml" --output "$out/eval_noise" --mode act
```

这些都是实际仿真或训练操作，无须为了阅读本报告立即重跑。每个运行目录内保留轨迹、结果、初始化记录、脚本/配置及输入哈希。
