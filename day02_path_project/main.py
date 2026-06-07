import json
from pathlib import Path


AIGC_TAGS = {
    "diffusion",
    "CLIP",
    "DiT",
    "multimodal",
    "text-to-image",
    "image-editing",
}


def load_prompts(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_aigc_related(prompt_item):
    tags = set(prompt_item["tags"])
    return bool(tags & AIGC_TAGS)


def select_prompts(prompts):
    selected = []
    for item in prompts:
        if is_aigc_related(item):
            selected.append(item)
    return selected


def save_prompts(prompts, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)


def main():
    project_dir = Path(__file__).parent
    input_path = project_dir / "data" / "prompts.json"
    output_path = project_dir / "outputs" / "selected_prompts.json"

    prompts = load_prompts(input_path)
    selected_prompts = select_prompts(prompts)
    save_prompts(selected_prompts, output_path)

    print(f"读取 prompt 总数：{len(prompts)}")
    print(f"筛选命中数量：{len(selected_prompts)}")
    for item in selected_prompts:
        print(item["id"], item["tags"])
    print(f"输出文件：{output_path}")


if __name__ == "__main__":
    main()
