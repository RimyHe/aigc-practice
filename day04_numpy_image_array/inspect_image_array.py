import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb_array(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    return np.array(raw_data)


def load_image_array(image_path):
    image = Image.open(image_path).convert("RGB")
    return np.array(image)


def build_summary(arr, image_arr, arr_norm, arr_chw):
    pixel_y = 0
    pixel_x = 0
    pixel = arr[pixel_y, pixel_x]
    image_pixel = image_arr[pixel_y, pixel_x]
    channel_means = arr_norm.mean(axis=(0, 1))
    dominant_index = int(np.argmax(channel_means))
    dominant_channel = ["R", "G", "B"][dominant_index]
    image_matches_json = bool(np.array_equal(arr, image_arr))

    lines = [
        "NumPy 图像数组检查结果",
        "=" * 28,
        "一、从 JSON 读取的像素矩阵",
        f"原始数组 shape：{arr.shape}",
        f"原始数组 dtype：{arr.dtype}",
        f"原始数组最小值：{arr.min()}",
        f"原始数组最大值：{arr.max()}",
        "",
        f"示例像素位置：(y={pixel_y}, x={pixel_x})",
        f"示例像素 RGB：R={pixel[0]}, G={pixel[1]}, B={pixel[2]}",
        "解释：这个像素是纯红色，因为 R 很高，G 和 B 都是 0。",
        "",
        "二、从 PNG 图片读取的像素矩阵",
        f"图片数组 shape：{image_arr.shape}",
        f"图片数组 dtype：{image_arr.dtype}",
        f"图片数组最小值：{image_arr.min()}",
        f"图片数组最大值：{image_arr.max()}",
        f"图片同一位置 RGB：R={image_pixel[0]}, G={image_pixel[1]}, B={image_pixel[2]}",
        f"PNG 图片数组是否和 JSON 数组完全一致：{image_matches_json}",
        "解释：图片文件被打开后，也会变成高度、宽度、RGB 通道组成的数字矩阵。",
        "",
        "三、归一化和通道顺序转换",
        f"归一化后最小值：{arr_norm.min():.3f}",
        f"归一化后最大值：{arr_norm.max():.3f}",
        "",
        "每个通道的平均亮度，范围是 0 到 1：",
        f"- R 平均亮度：{channel_means[0]:.3f}",
        f"- G 平均亮度：{channel_means[1]:.3f}",
        f"- B 平均亮度：{channel_means[2]:.3f}",
        "",
        f"HWC shape：{arr_norm.shape}",
        f"CHW shape：{arr_chw.shape}",
        "解释：HWC 是高度、宽度、通道；CHW 是通道、高度、宽度。",
        "",
        f"平均亮度最高的通道：{dominant_channel}",
    ]
    return "\n".join(lines)


def save_summary(summary_text, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary_text)


def main():
    project_dir = Path(__file__).parent
    input_path = project_dir / "data" / "tiny_rgb_array.json"
    image_path = project_dir / "data" / "tiny_rgb_image.png"
    output_path = project_dir / "outputs" / "array_summary.txt"

    arr = load_rgb_array(input_path)
    image_arr = load_image_array(image_path)
    arr_norm = arr / 255.0
    arr_chw = arr_norm.transpose(2, 0, 1)

    summary_text = build_summary(arr, image_arr, arr_norm, arr_chw)
    save_summary(summary_text, output_path)

    print(summary_text)
    print()
    print(f"统计结果已保存到：{output_path}")


if __name__ == "__main__":
    main()
