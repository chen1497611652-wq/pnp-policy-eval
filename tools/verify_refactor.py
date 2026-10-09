"""D14 重构验收: 静态扫描 + 运行结果逐位对比。

在仓库根目录运行(必须用装了依赖的那个解释器):

    & "D:\\miniconda3-1\\envs\\py310\\python.exe" tools\\verify_refactor.py

这个脚本做第 2、3 层:

  第 2 层 静态扫描  src/evaluate.py 的未定义名 / 死配置 / 语法
  第 3 层 运行对比  基线与重构的 results.jsonl 逐位比较

第 1 层(源码 diff)不在这个脚本里 —— 它需要人读, 没法自动判定"哪些差异是预期的"。
在 Git Bash 里跑(COURSE = 课程仓库的 code/run_closed_loop.py):

    # a) 全文件: 配置块与文件头被整段重写(约 -121/+74)。这里才是要逐行看的地方 ——
    #    D14 当天卡住的 OUTPUT_DIR 就是在这段搬丢的。对着 YAML 逐组核:
    #    paths / dataset / policy / env / eval / video / runtime 都要有落点。
    diff <COURSE> src/evaluate.py

    # b) 函数体区域(require_layout 之后到文件尾): 预期恰好 4 个 hunk / 8 处改动
    #    ENV_XML x2、from src.video_io、ACTION_TYPE、STATE_TYPE、
    #    ENV_HZ、STATE_SLICE、load_config(parse_args().config)
    diff -u <(sed -n '/^def require_layout/,$p' <COURSE>) \
            <(sed -n '/^def require_layout/,$p' src/evaluate.py)

    # c) strict_snapshot() 一字未动 —— 它是 D15 的领地, 今天不许碰
    diff <(awk '/^def strict_snapshot/,/^def run_one_seed/' <COURSE>) \
         <(awk '/^def strict_snapshot/,/^def run_one_seed/' src/evaluate.py)   # 期望无输出

判定: 三层都通过, 才算证明了"重构没有改变行为"。
局限: 只覆盖配置里的那几个 seed、只覆盖本机本栈; 每次改动后要重跑。
      第 3 层全绿也不能单独当结论 —— 不在本次路径上的改动它看不见。
"""
from __future__ import annotations

import argparse
import ast
import builtins
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "src" / "evaluate.py"

# 天生会变、不应参与对比的字段: 墙钟耗时 / 输出绝对路径
VOLATILE = {"elapsed_s", "video"}

BUILTINS = set(dir(builtins))
FUNC_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


# ---------------------------------------------------------------- 第 2 层

def _bind_targets(node: ast.AST, found: set[str]) -> None:
    """收集赋值目标里的名字(支持 a, b = ... / a.b / a[b] 等)。"""
    if isinstance(node, ast.Name):
        found.add(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for item in node.elts:
            _bind_targets(item, found)
    elif isinstance(node, ast.Starred):
        _bind_targets(node.value, found)


def _collect_binds(node: ast.AST, found: set[str]) -> None:
    """收集 node 子树里所有绑定名。不进入嵌套函数体(那是独立作用域)。"""
    if isinstance(node, FUNC_NODES):
        found.add(node.name)
        return
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        for alias in node.names:
            found.add((alias.asname or alias.name).split(".")[0])
        return
    if isinstance(node, ast.Assign):
        for target in node.targets:
            _bind_targets(target, found)
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        _bind_targets(node.target, found)
    elif isinstance(node, ast.For):
        _bind_targets(node.target, found)
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        for item in node.items:
            if item.optional_vars is not None:
                _bind_targets(item.optional_vars, found)
    elif isinstance(node, ast.Try):
        for handler in node.handlers:
            if handler.name:
                found.add(handler.name)
    elif isinstance(node, ast.Global):
        found.update(node.names)
    elif isinstance(node, ast.NamedExpr):
        _bind_targets(node.target, found)
    elif isinstance(node, ast.comprehension):
        _bind_targets(node.target, found)

    for child in ast.iter_child_nodes(node):
        if isinstance(child, FUNC_NODES):
            found.add(child.name)
            continue
        _collect_binds(child, found)


def _loads(nodes: list[ast.AST]) -> list[ast.Name]:
    """nodes 子树里(不进入嵌套函数体)所有 Name 的读取。"""
    found: list[ast.Name] = []

    def walk(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, FUNC_NODES):
                continue
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                found.append(child)
            walk(child)

    for node in nodes:
        if isinstance(node, FUNC_NODES):
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            found.append(node)
        walk(node)
    return found


def layer_static(source: Path) -> list[str]:
    """第 2 层: 未定义名 + 死配置。返回问题列表(空=通过)。"""
    text = source.read_text(encoding="utf-8")
    tree = ast.parse(text)                     # 语法错会在这里抛 SyntaxError
    problems: list[str] = []

    module_names: set[str] = set()
    for node in tree.body:
        _collect_binds(node, module_names)

    # __file__ 是模块自带属性, 不是未定义名
    ignored = BUILTINS | {"__file__"}
    for name in _loads(tree.body):
        if name.id not in ignored and name.id not in module_names:
            problems.append(f"{source.name}:{name.lineno}  [<module>]  未定义名 {name.id}")

    for func in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        local: set[str] = {a.arg for a in func.args.args + func.args.kwonlyargs + func.args.posonlyargs}
        if func.args.vararg:
            local.add(func.args.vararg.arg)
        if func.args.kwarg:
            local.add(func.args.kwarg.arg)
        for node in func.body:
            _collect_binds(node, local)
        declared_global = {n for s in ast.walk(func) if isinstance(s, ast.Global) for n in s.names}
        visible = local | module_names | ignored | declared_global
        for name in _loads(func.body):
            if name.id not in visible:
                problems.append(
                    f"{source.name}:{name.lineno}  [{func.name}()]  未定义名 {name.id}"
                )

    # 死配置: 模块级赋值了、但全文从没被读取过
    read_names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    module_assigned: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                _bind_targets(target, module_assigned)
    dead = sorted(n for n in module_assigned if n not in read_names)
    if dead:
        problems.append("死配置(模块级赋值但从未被读取): " + ", ".join(dead))

    return problems


# ---------------------------------------------------------------- 第 3 层

def _load_rows(path: Path) -> dict[int, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {int(row["seed"]): row for row in rows}


def layer_compare(baseline_path: Path, refactor_path: Path) -> tuple[list[str], int, int]:
    """第 3 层: 除 VOLATILE 外, 所有字段逐位比较(浮点按精确值比较)。"""
    baseline = _load_rows(baseline_path)
    refactor = _load_rows(refactor_path)
    problems: list[str] = []

    if len(baseline) != len(refactor):
        problems.append(f"行数不同: {len(baseline)} vs {len(refactor)}")

    seeds = sorted(set(baseline) | set(refactor))
    keys = set()
    for row in list(baseline.values()) + list(refactor.values()):
        keys |= set(row)
    keys -= VOLATILE

    for seed in seeds:
        if seed not in baseline or seed not in refactor:
            problems.append(f"seed {seed}: 只存在于一边")
            continue
        for key in sorted(keys):
            left, right = baseline[seed].get(key), refactor[seed].get(key)
            # repr 一起比: 能区分 0.0 与 -0.0, 也保证浮点是"逐位"而非"近似"
            if not (left == right and repr(left) == repr(right)):
                problems.append(f"seed {seed}  {key}: {left!r} -> {right!r}")
    return problems, len(keys), len(seeds)


# ---------------------------------------------------------------- main

def _default_results(config_path: Path, baseline: bool = False) -> Path:
    """从 YAML 推默认结果路径, 避免把绝对路径写死进代码。

    重构版: <output_root>/<type>_closed_loop/results.jsonl
    基线版: <output_root 的上一级>/_d14_baseline/<type>_closed_loop/results.jsonl
            —— D14 基线是把 OUTPUT_ROOT 指到 outputs/_d14_baseline 跑的, 跟 outputs/d14 同级
    """
    import yaml

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = Path(cfg["paths"]["output_root"]).expanduser()
    if baseline:
        output_root = output_root.parent / "_d14_baseline"
    policy_type = cfg["policy"]["type"]
    return output_root / f"{policy_type}_closed_loop" / "results.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description="D14 重构验收")
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "default.yaml")
    parser.add_argument("--source", type=Path, default=SOURCE,
                        help="要静态扫描的文件(默认 src/evaluate.py)")
    parser.add_argument("--baseline-results", type=Path,
                        help="基线结果 (默认: output_root 上一级的 _d14_baseline/ 下)")
    parser.add_argument("--results", type=Path,
                        help="重构版结果 (默认: output_root 下)")
    parser.add_argument("--static-only", action="store_true", help="只跑第 2 层")
    args = parser.parse_args()

    ok = True

    print("=" * 66)
    print("第 2 层 | 静态扫描:", args.source)
    print("=" * 66)
    try:
        problems = layer_static(args.source)
    except SyntaxError as exc:
        print(f"  [FAIL] 语法错误: {exc}")
        return 1
    if problems:
        ok = False
        for line in problems:
            print("  [FAIL]", line)
    else:
        print("  [PASS] 无未定义名, 无死配置, 语法可编译")

    if args.static_only:
        return 0 if ok else 1

    baseline_path = args.baseline_results or _default_results(args.config, baseline=True)
    refactor_path = args.results or _default_results(args.config)

    print()
    print("=" * 66)
    print("第 3 层 | 运行对比(逐位)")
    print("=" * 66)
    print(f"  基线: {baseline_path}")
    print(f"  重构: {refactor_path}")
    for path in (baseline_path, refactor_path):
        if not path.exists():
            print(f"  [FAIL] 找不到: {path}")
            return 1

    problems, n_keys, n_seeds = layer_compare(baseline_path, refactor_path)
    if problems:
        ok = False
        for line in problems:
            print("  [FAIL]", line)
    else:
        print(f"  [PASS] {n_keys} 个字段 x {n_seeds} 个 seed 全部逐位相同")
        print(f"         (已排除 {', '.join(sorted(VOLATILE))} —— 墙钟与输出路径天生会变)")

    print()
    print("=" * 66)
    print("验收结果:", "通过" if ok else "未通过")
    print("=" * 66)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
