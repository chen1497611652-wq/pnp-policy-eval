"""不可复现性检验: 判断同一份评测在不同次执行之间是否逐位一致。

用法:
    python -m src.nondeterminism_check <结果文件> [<结果文件> ...] [--ignore f1,f2]

输入文件可以是 results.jsonl, 也可以是混着日志行的 run.log
(只解析能 json.loads 成功的行, 其余跳过)。
忽略字段默认 elapsed_s,video —— 墙钟和输出路径天生会变, 不该参与对比。
"""
import argparse
import json
import sys
from pathlib import Path

DEFAULT_IGNORE = ("elapsed_s", "video")

# 「结局类」字段: 这几个一旦不同, 不是抖动, 是结论翻面
OUTCOME_KEYS = ("physical_success", "legacy_success", "action_steps")
# 退出码: 「没法判定」必须是独立的非零码, 否则在 CI 里会伪装成通过
EXIT_OK = 0
EXIT_NONDET = 1
EXIT_UNKNOWN = 2


def unique_labels(paths: list[Path]) -> list[str]:
    """给每个文件取一个尽量短的标签: 从右往左多取一层, 直到彼此不重名。"""
    for k in range(2, 10):
        labels = ["/".join(p.parts[-k:]) for p in paths]
        if len(set(labels)) == len(labels):
            return labels
    return [str(p) for p in paths]

def load_rows(path: Path, label: str) -> tuple[list[tuple[str, dict]], int]:
    """读一个文件里的所有结果行, 返回 (行列表, 跳过的行数)。

    每行存成 (文件名, 那一行的字典) —— 来源要一路带到报告里,
    否则报差异时说不清"哪个值来自哪个文件"。
    """
    rows: list[tuple[str, dict]] = []
    skipped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if not isinstance(row, dict) or "seed" not in row:
            skipped += 1
            continue
        rows.append((label, row))
    return rows, skipped

def group_by_seed(rows:list[tuple[str, dict]]) -> dict[int, list[tuple[str,dict]]]:
    """把所有文件的每一行按 seed 归组。

    注意: 同一个文件里同一个 seed 出现两次也算两次 —— D10 的验收标准
    就是「连跑两次写进同一个 results.jsonl, action_steps 必须一模一样」。
    """
    groups: dict[int, list[tuple[str, dict]]] = {}
    for source, row in rows:
        seed = int(row["seed"])
        groups.setdefault(seed, []).append((source, row))
    return groups

class _Missing:
    """字段缺失的哨兵。不能用 None —— None 是合法的 JSON 值, 会混。"""
    def __repr__(self) -> str:
        return "<缺失>"


MISSING = _Missing()


def compare_group(group: list[tuple[str, dict]],
                  ignore: set[str]) -> list[tuple[str, bool]]:
    """组内逐位比较, 返回 [(差异描述, 是否结局类字段), ...]; 空列表 = 这一组通过。

    返回的 bool 让 verdict 不用去猜哪个字段是结局类。
    """
    fields: set[str] = set()
    for _, row in group:
        fields |= set(row)
    fields -= set(ignore)

    diffs: list[tuple[str, bool]] = []
    for field in sorted(fields):
        seen: dict[str, list[str]] = {}
        for source, row in group:
            value = row[field] if field in row else MISSING
            seen.setdefault(repr(value), []).append(source)
        if len(seen) == 1:          # 只要一种值 → 逐位一致
            continue

        values = [repr(row[field]) if field in row else repr(MISSING)
                  for _, row in group]
        sep = "  ->  " if len(values) == 2 else "  |  "
        diffs.append((f"{field}: {sep.join(values)}", field in OUTCOME_KEYS))
    return diffs

def verdict(seed: int, group: list[tuple[str, dict]],
            diffs: list[tuple[str, bool]]) -> tuple[str, list[str]]:
    """判一个 seed。返回 (等级, 要打印的行); 等级 ∈ {"OK", "NONDET", "??"}。"""
    n = len(group)
    if n < 2:
        return "??", [f"[??]     seed {seed:>4}: 只有 {n} 次执行, 没有可比对象"]

    if not diffs:
        return "OK", [f"[OK]     seed {seed:>4}: {n} 次执行逐位一致"]

    outcome = [t for t, is_out in diffs if is_out]
    note = (f", 其中 {len(outcome)} 处是结局类字段 —— 不是抖动, 是结论翻面"
            if outcome else "  (全是数值抖动, 结论未翻面)")
    lines = [f"[NONDET] seed {seed:>4}: {len(diffs)} 处差异{note}"]
    lines += [f"            {'⚠️ ' if is_out else '   '}{t}" for t, is_out in diffs]
    return "NONDET", lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="不可复现性检验: 同一 seed 的多次执行是否逐位一致")
    parser.add_argument("files", type=Path, nargs="+",
                        help="两份以上的结果文件(results.jsonl 或 run.log)")
    parser.add_argument("--ignore", default=",".join(DEFAULT_IGNORE),
                        help="不参与对比的字段, 逗号分隔(默认 elapsed_s,video)")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    labels = unique_labels(args.files)
    absent = [str(f) for f in args.files if not f.is_file()]
    if absent:
        print("找不到这些文件:\n  " + "\n  ".join(absent), file=sys.stderr)
        sys.exit(EXIT_UNKNOWN)
    ignored = {f.strip() for f in args.ignore.split(",") if f.strip()}

    all_rows: list[tuple[str, dict]] = []
    sources: list[tuple[str, list[int], int]] = []
    for f, lab in zip(args.files, labels):
        rows, skipped = load_rows(f, lab)
        sources.append((lab, [int(r["seed"]) for _, r in rows], skipped))
        all_rows.extend(rows)

    if not all_rows:
        print("没有读到任何结果行: 确认文件是 results.jsonl 或 run.log, 且非空。",
              file=sys.stderr)
        sys.exit(EXIT_UNKNOWN)

    groups = group_by_seed(all_rows)

    fields: set[str] = set()
    for group in groups.values():
        for _, row in group:
            fields |= set(row)
    fields -= ignored

    print(f"口径: {len(args.files)} 份文件 / {len(groups)} 个 seed / "
          f"每次比较 {len(fields)} 个字段   (忽略: {', '.join(sorted(ignored))})")
    for lab, seeds, skipped in sources:
        span = f"seed {min(seeds)}-{max(seeds)}" if seeds else "无结果行"
        print(f"  {lab}: {len(seeds)} 行 [{span}], 跳过 {skipped} 行")
    print()

    levels: list[str] = []
    for seed, group in sorted(groups.items()):
        level, lines = verdict(seed, group, compare_group(group, ignored))
        levels.append(level)
        for line in lines:
            print(line)

    print()
    print(f"合计: {levels.count('OK')} 一致 / {levels.count('NONDET')} 有差异 / "
          f"{levels.count('??')} 无法判定")
    if "NONDET" in levels:
        sys.exit(EXIT_NONDET)
    if "??" in levels:
        sys.exit(EXIT_UNKNOWN)
    sys.exit(EXIT_OK)


if __name__ == "__main__":
    main()
