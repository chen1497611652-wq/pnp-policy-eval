"""有限微调试验：重复观测保留末条；不改原数据、动作时间轴或归一化统计。"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import src.evaluate as ev


def last_observation_indices(states, encoded_images):
    """相同PNG字节和float32状态的观测只保留最后出现的索引。"""
    last = {}
    for i, (state, image) in enumerate(zip(states, encoded_images)):
        key = (np.asarray(state, dtype=np.float32).tobytes(), hashlib.sha256(image).digest())
        last[key] = i
    return sorted(last.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    ev.load_config(args.config)
    if args.output.exists():
        raise FileExistsError("Candidate output exists")
    args.output.mkdir(parents=True)
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("Positive training budget required")
    import pandas as pd
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    source = sorted((ev.DATASET_ROOT/"data").rglob("*.parquet"))[0]
    frame = pd.read_parquet(source, columns=["observation.state", "observation.image"])
    states = np.stack(frame["observation.state"])
    indices = last_observation_indices(states, [v["bytes"] for v in frame["observation.image"]])
    (args.output/"selected_indices.json").write_text(json.dumps(indices), encoding="utf-8")
    torch.manual_seed(6); np.random.seed(6)
    policy = ev.load_policy(ev.POLICY_TYPE, Path(ev.POLICY_PATH))
    horizon = policy.config.chunk_size
    info = json.loads((ev.DATASET_ROOT/"meta/info.json").read_text(encoding="utf-8"))
    dataset = LeRobotDataset(ev.DATASET_REPO_ID, root=str(ev.DATASET_ROOT), image_transforms=None,
                            delta_timestamps={"action":[i/info["fps"] for i in range(horizon)]})
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=True,
                        num_workers=0, drop_last=True, generator=torch.Generator().manual_seed(6))
    stats_before = {k:v.detach().cpu().clone() for k,v in policy.state_dict().items()
                    if k.startswith(("normalize_inputs.", "normalize_targets.", "unnormalize_outputs."))}
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5, weight_decay=1e-4)
    policy.train()
    wanted = [*policy.config.input_features, "action", "action_is_pad"]
    print(f"selected {len(indices)}/{len(frame)} frames; steps={args.steps}, batch={args.batch_size}", flush=True)
    iterator = iter(loader)
    start = time.perf_counter()
    with (args.output/"training.jsonl").open("x", encoding="utf-8") as log:
        for step in range(args.steps):
            try:
                sample = next(iterator)
            except StopIteration:
                iterator = iter(loader); sample = next(iterator)
            batch = {k:sample[k].to(ev.DEVICE) for k in wanted}
            optimizer.zero_grad(set_to_none=True)
            loss, details = policy.forward(batch)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
            optimizer.step()
            row = {"step":step+1, "loss":float(loss.detach()), "grad_norm":float(grad_norm),
                   "l1":float(details.get("l1_loss", float("nan"))), "kld":float(details.get("kld_loss", float("nan")))}
            log.write(json.dumps(row)+"\n")
            if (step+1)%25 == 0 or step == 0:
                log.flush()
                print(f"step {step+1}/{args.steps} loss={row['loss']:.6f} elapsed={time.perf_counter()-start:.1f}s", flush=True)
    policy.eval()
    state_dict = policy.state_dict()
    for key, value in stats_before.items():
        assert torch.equal(value, state_dict[key].detach().cpu()), key
    checkpoint = args.output/"checkpoint"
    policy.save_pretrained(checkpoint)
    config = __import__("yaml").safe_load(args.config.read_text(encoding="utf-8"))
    config["policy"]["path"] = str(checkpoint.resolve())
    config["paths"]["output_root"] = str(args.output/"evaluation")
    (args.output/"candidate.yaml").write_text(__import__("yaml").safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    report = {"source_frames":len(frame), "selected_frames":len(indices), "optimizer_steps":args.steps,
              "batch_size":args.batch_size, "lr":1e-5, "seed":6, "seconds":time.perf_counter()-start,
              "normalization_unchanged":True,
              "sampling":"Last occurrence of identical PNG bytes + float32 state; targets remain original i:i+chunk_size, no retiming",
              "training":"Original ACT variational training forward; labels only used during training, not evaluation",
              "scope":"Exploratory candidate, not a proven fix; additional training and sampling both change compared with original checkpoint",
              "files":{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       [source, Path(ev.POLICY_PATH)/"model.safetensors", args.config, Path(__file__)]}}
    (args.output/"training_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    shutil.copy2(Path(__file__), args.output/Path(__file__).name)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
