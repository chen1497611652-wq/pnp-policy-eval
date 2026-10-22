from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "results" / "smolvla_d15_seeds6-11.jsonl"
INCLUDE_IMAGES = True


def load_records(path):
    records = []
    required = ("seed", "physical_success", "max_target_lift", "upright_cos")

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"第 {line_number} 行不是合法 JSON"
            ) from error

        if not isinstance(record, dict):
            raise ValueError(f"第 {line_number} 行必须是一条字典记录")

        for field in required:
            if field not in record:
                raise ValueError(
                    f"第 {line_number} 行缺少字段：{field}"
                )

        if type(record["seed"]) is not int:
            raise ValueError(f"第 {line_number} 行的 seed 必须是整数")

        if type(record["physical_success"]) is not bool:
            raise ValueError(
                f"第 {line_number} 行的 physical_success 必须是 true 或 false"
            )

        for field in ("max_target_lift", "upright_cos"):
            if type(record[field]) not in (int, float):
                raise ValueError(
                    f"第 {line_number} 行的 {field} 必须是数字"
                )

        records.append(record)

    if not records:
        raise ValueError("输入文件没有有效记录")

    return records


def make_report(records, source_path, images):
    success_count = 0

    for record in records:
        if record["physical_success"]:
            success_count += 1

    total_count = len(records)
    success_rate = success_count / total_count

    lines = [
        "# 评测报告",
        "",
        f"数据来源：`{source_path.name}`",
        "",
        f"本批共评测 {total_count} 次，严格成功 {success_count} 次，成功率为 {success_rate:.2%}。",
        "",
        "| Seed | 严格成功 | 最大抬升高度（cm） | 最终直立度 |",
        "|---|---|---|---|",
    ]

    for record in records:
        result = "成功" if record["physical_success"] else "失败"
        lift_cm = record["max_target_lift"] * 100
        upright = record["upright_cos"]

        lines.append(
            f"| {record['seed']} | {result} | {lift_cm:.2f} | {upright:.4f} |"
        )

    lines.extend(["", "## 图表", ""])

    if not images:
        lines.append("本次未附图。")

    for title, image_path in images:
        lines.extend([f"![{title}]({image_path.name})", ""])

    lines.extend([
        "",
        "## 结论范围",
        "",
        "本报告仅描述所选记录，不能据此确定模型的整体成功率或失败的根本原因。",
        "最大抬升高度是整个过程的最大值，最终直立度是结束时的值，两者不一定来自同一时刻。",
    ])

    return "\n".join(lines) + "\n"


def main():
    records = load_records(INPUT_PATH)

    images = []

    if INCLUDE_IMAGES:
        for title, suffix in [
            ("本批严格成功率", "success_rate"),
            ("最大抬升高度", "max_lift"),
            ("最终直立度", "uprightness"),
        ]:
            image_path = INPUT_PATH.with_name(
                f"{INPUT_PATH.stem}_{suffix}.png"
            )

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"缺少本批图片：{image_path.name}"
                )

            images.append((title, image_path))

    report = make_report(records, INPUT_PATH, images)

    output_path = INPUT_PATH.with_name(
        f"{INPUT_PATH.stem}_report.md"
    )
    output_path.write_text(report, encoding="utf-8")

    print("报告已保存：", output_path.resolve())


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print("生成报告失败：", error)
        raise SystemExit(1)