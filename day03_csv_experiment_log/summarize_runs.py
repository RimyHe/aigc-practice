import csv
from pathlib import Path


def load_runs(csv_path):
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        runs = []
        for row in reader:
            row["seed"] = int(row["seed"])
            row["guidance_scale"] = float(row["guidance_scale"])
            row["steps"] = int(row["steps"])
            row["clip_score"] = float(row["clip_score"])
            row["human_rating"] = float(row["human_rating"])
            runs.append(row)
    return runs


def count_by_model(runs):
    model_counts = {}
    for run in runs:
        model = run["model"]
        model_counts[model] = model_counts.get(model, 0) + 1
    return model_counts


def build_summary(runs):
    total_runs = len(runs)
    avg_clip_score = sum(run["clip_score"] for run in runs) / total_runs
    avg_human_rating = sum(run["human_rating"] for run in runs) / total_runs
    best_run = max(runs, key=lambda run: run["human_rating"])
    model_counts = count_by_model(runs)
    highest_guidance_run = max(runs, key=lambda run: run["guidance_scale"])
    high_clip_runs = [run for run in runs if run["clip_score"] > 0.30]
    
    lines = [
        "AIGC 生成实验统计结果",
        "=" * 28,
        f"总实验数：{total_runs}",
        f"平均 CLIP 分数：{avg_clip_score:.3f}",
        f"平均人工评分：{avg_human_rating:.2f}",
        "",
        "人工评分最高的实验：",
        f"- run_id：{best_run['run_id']}",
        f"- model：{best_run['model']}",
        f"- prompt：{best_run['prompt']}",
        f"- seed：{best_run['seed']}",
        f"- guidance_scale：{best_run['guidance_scale']}",
        f"- steps：{best_run['steps']}",
        f"- clip_score：{best_run['clip_score']:.3f}",
        f"- human_rating：{best_run['human_rating']:.1f}",
        f"- notes：{best_run['notes']}",
        "",
        "guidance_scale 最高的实验：",
        f"- run_id：{highest_guidance_run['run_id']}",
        f"- guidance_scale：{highest_guidance_run['guidance_scale']}",
        "",
        f"CLIP 分数大于 0.30 的实验数：{len(high_clip_runs)}",
        "",
        "按模型统计实验数量：",
    ]

    for model, count in model_counts.items():
        lines.append(f"- {model}：{count} 次")

    return "\n".join(lines)


def save_summary(summary_text, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary_text)


def main():
    project_dir = Path(__file__).parent
    input_path = project_dir / "data" / "generation_runs.csv"
    output_path = project_dir / "outputs" / "summary.txt"

    runs = load_runs(input_path)
    summary_text = build_summary(runs)
    save_summary(summary_text, output_path)

    print(summary_text)
    print()
    print(f"统计结果已保存到：{output_path}")


if __name__ == "__main__":
    main()
