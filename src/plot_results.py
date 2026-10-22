from pathlib import Path
import json
import matplotlib.pyplot as plt

path = Path("results") / "smolvla_d15_seeds6-11.jsonl"

text = path.read_text(encoding="utf-8")
lines = text.splitlines()

print("记录行数：", len(lines))
print("第一行：", lines[0])

records = []

for line in lines:
    record = json.loads(line)
    records.append(record)

print("收集到的记录数：", len(records))
print("第一条记录的seed: ",records[0]["seed"])

success_count = 0

for record in records:
    if record["physical_success"]:
        success_count += 1

total_count =len(records)
success_rate = success_count /total_count

print("成功次数: ", success_count)
print("总次数: ", total_count)
print(f"本批成功率: {success_rate:.2%}")

fig, ax = plt.subplots(figsize=(5,4))

ax.bar(["SmolVLA"], [success_rate*100],width=0.5)

ax.set_ylim(0,100)
ax.set_ylabel("Strict success rate (%)")
ax.set_title("SmolVLA: seeds 6-11")

ax.text(
    0,
    success_rate*100+2,
    f"{success_count}/{total_count} ({success_rate:.2%})",
    ha="center",
)

plt.tight_layout()

output_path = path.parent / "smolvla_d15_seeds6-11_success_rate.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight")

print("图片已保存：",output_path.resolve())


seeds = []
lift_heights_cm = []

for record in records:
    seeds.append(str(record["seed"]))
    lift_heights_cm.append(record["max_target_lift"] * 100)

fig_lift, ax_lift = plt.subplots(figsize=(7, 4))

ax_lift.bar(seeds, lift_heights_cm, color="steelblue")

ax_lift.set_xlabel("Seed")
ax_lift.set_ylabel("Maximum mug lift (cm)")
ax_lift.set_title("SmolVLA: seeds 6-11")

ax_lift.axhline(
    y=3,
    color="red",
    linestyle="--",
    label="Lift threshold: 3 cm",
)
ax_lift.legend()

fig_lift.tight_layout()

lift_output_path = path.parent / "smolvla_d15_seeds6-11_max_lift.png"
fig_lift.savefig(lift_output_path, dpi=150, bbox_inches="tight")

print("抬升高度图已保存：", lift_output_path.resolve())


upright_values = []

for record in records:
    upright_values.append(record["upright_cos"])

fig_upright, ax_upright = plt.subplots(figsize=(7, 4))

ax_upright.bar(seeds, upright_values, color="steelblue")

ax_upright.set_xlabel("Seed")
ax_upright.set_ylabel("Final uprightness (cosine)")
ax_upright.set_title("SmolVLA: seeds 6-11")
ax_upright.set_ylim(-1, 1.1)

ax_upright.axhline(
    y=0.7,
    color="red",
    linestyle="--",
    label="Uprightness threshold: 0.7",
)

ax_upright.legend(loc="lower right")
fig_upright.tight_layout()

upright_output_path = path.parent / "smolvla_d15_seeds6-11_uprightness.png"
fig_upright.savefig(upright_output_path, dpi=150, bbox_inches="tight")

print("最终直立度图已保存：", upright_output_path.resolve())

plt.show()
