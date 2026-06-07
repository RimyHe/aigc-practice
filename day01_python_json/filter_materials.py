import json
from pathlib import Path


KEYWORDS = {"diffusion", "CLIP", "image generation", "multimodal"}


def load_materials(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_related_to_aigc(item):
    tags = set(item["keywords"])
    return bool(tags & KEYWORDS)


def main():
    current_dir = Path(__file__).parent
    json_path = current_dir / "materials.json"

    materials = load_materials(json_path)
    matched = []

    for item in materials:
        if is_related_to_aigc(item):
            matched.append(item)

    print("AIGC / 多模态相关学习材料：")
    print("-" * 40)

    for index, item in enumerate(matched, start=1):
        print(f"{index}. {item['title']}")
        print(f"   方向：{item['direction']}")
        print(f"   备注：{item['note']}")
        print()


if __name__ == "__main__":
    main()
