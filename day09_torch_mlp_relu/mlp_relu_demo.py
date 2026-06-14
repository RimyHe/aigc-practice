from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image, ImageDraw


CLASS_NAMES = ["same_sign", "different_sign"]
CLASS_COLORS = [
    (230, 90, 90),
    (80, 130, 230),
]


def make_xor_dataset(samples_per_cluster=50):
    torch.manual_seed(0)
    centers = torch.tensor(
        [
            [-2.0, -2.0],
            [2.0, 2.0],
            [-2.0, 2.0],
            [2.0, -2.0],
        ]
    )
    labels_for_centers = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    features = []
    labels = []
    for center, label in zip(centers, labels_for_centers):
        points = center + torch.randn(samples_per_cluster, 2) * 0.45
        class_labels = torch.full((samples_per_cluster,), label.item(), dtype=torch.long)
        features.append(points)
        labels.append(class_labels)

    x = torch.cat(features, dim=0)
    y = torch.cat(labels, dim=0)
    return x, y


def make_dataloader(x, y, batch_size=16):
    dataset = TensorDataset(x, y)
    generator = torch.Generator().manual_seed(0)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    return dataset, dataloader


def build_linear_model():
    torch.manual_seed(1)
    return nn.Linear(2, 2)


def build_mlp_model(hidden_size=16):
    torch.manual_seed(1)
    return nn.Sequential(
        nn.Linear(2, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 2),
    )


def evaluate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            logits = model(batch_x)
            predicted = logits.argmax(dim=1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.numel()
    return correct / total


def train_model(model, dataloader, model_name, num_epochs=200, lr=0.03):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.08)
    dataset_size = len(dataloader.dataset)
    log_lines = [
        f"{model_name} 训练日志",
        "=" * 36,
        f"optimizer：Adam",
        f"learning_rate：{lr}",
        f"epochs：{num_epochs}",
        "",
        "epoch | avg_loss | accuracy",
    ]

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0

        model.train()
        for batch_x, batch_y in dataloader:
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.shape[0]

        avg_loss = total_loss / dataset_size
        accuracy = evaluate_accuracy(model, dataloader)
        log_line = f"{epoch:>5} | {avg_loss:>8.5f} | {accuracy:>8.3f}"
        log_lines.append(log_line)

        if epoch in {1, 2, 3, 5, 10, 20, 50, 100, 150, num_epochs}:
            print(f"{model_name} {log_line}")

    return "\n".join(log_lines)


def describe_samples(model, x, y, title):
    sample_indices = [0, 50, 100, 150, 25, 125]
    model.eval()
    with torch.no_grad():
        sample_x = x[sample_indices]
        sample_y = y[sample_indices]
        logits = model(sample_x)
        probs = torch.softmax(logits, dim=1)
        predicted = probs.argmax(dim=1)

    lines = [title]
    for row, original_index in enumerate(sample_indices):
        lines.append(
            f"sample {original_index}: true={CLASS_NAMES[sample_y[row]]}, "
            f"logits={logits[row].tolist()}, "
            f"probs={[round(v, 4) for v in probs[row].tolist()]}, "
            f"pred={CLASS_NAMES[predicted[row]]}"
        )
    return "\n".join(lines)


def save_decision_map(model, x, y, output_path, title):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = 480
    height = 480
    x_min, x_max = -4.0, 4.0
    y_min, y_max = -4.0, 4.0

    image = Image.new("RGB", (width, height), "white")
    pixels = image.load()

    model.eval()
    with torch.no_grad():
        for py in range(height):
            coord_y = y_max - (py / (height - 1)) * (y_max - y_min)
            row_points = []
            for px in range(width):
                coord_x = x_min + (px / (width - 1)) * (x_max - x_min)
                row_points.append([coord_x, coord_y])

            grid_tensor = torch.tensor(row_points, dtype=torch.float32)
            predicted = model(grid_tensor).argmax(dim=1)

            for px, class_id in enumerate(predicted.tolist()):
                base_color = CLASS_COLORS[class_id]
                pixels[px, py] = tuple(int(0.82 * 255 + 0.18 * c) for c in base_color)

    draw = ImageDraw.Draw(image)

    def to_pixel(point):
        px = int((point[0] - x_min) / (x_max - x_min) * (width - 1))
        py = int((y_max - point[1]) / (y_max - y_min) * (height - 1))
        return px, py

    for point, label in zip(x, y):
        px, py = to_pixel(point.tolist())
        color = CLASS_COLORS[label.item()]
        radius = 5
        draw.ellipse(
            (px - radius, py - radius, px + radius, py + radius),
            fill=color,
            outline=(30, 30, 30),
        )

    axis_color = (120, 120, 120)
    zero_x, zero_y = to_pixel([0.0, 0.0])
    draw.line((zero_x, 0, zero_x, height), fill=axis_color, width=1)
    draw.line((0, zero_y, width, zero_y), fill=axis_color, width=1)
    draw.text((12, 12), title, fill=(0, 0, 0))

    for i, name in enumerate(CLASS_NAMES):
        y0 = 38 + i * 22
        draw.rectangle((12, y0, 26, y0 + 14), fill=CLASS_COLORS[i])
        draw.text((34, y0 - 2), name, fill=(0, 0, 0))

    image.save(output_path)


def save_log(log_text, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(log_text)


def main():
    project_dir = Path(__file__).parent
    output_dir = project_dir / "outputs"
    log_path = output_dir / "training_log.txt"
    linear_map_path = output_dir / "linear_boundary.png"
    mlp_map_path = output_dir / "mlp_boundary.png"

    x, y = make_xor_dataset(samples_per_cluster=50)
    dataset, dataloader = make_dataloader(x, y, batch_size=16)

    print(f"样本总数：{len(dataset)}")
    print(f"x shape：{tuple(x.shape)}")
    print(f"y shape：{tuple(y.shape)}")
    print()

    linear_model = build_linear_model()
    mlp_model = build_mlp_model(hidden_size=16)

    linear_log = train_model(linear_model, dataloader, "Linear baseline", num_epochs=200, lr=0.003)
    print()
    mlp_log = train_model(mlp_model, dataloader, "MLP + ReLU", num_epochs=200, lr=0.03)

    linear_acc = evaluate_accuracy(linear_model, dataloader)
    mlp_acc = evaluate_accuracy(mlp_model, dataloader)
    linear_samples = describe_samples(linear_model, x, y, "Linear baseline 样本预测：")
    mlp_samples = describe_samples(mlp_model, x, y, "MLP + ReLU 样本预测：")

    save_decision_map(linear_model, x, y, linear_map_path, "Linear baseline")
    save_decision_map(mlp_model, x, y, mlp_map_path, "MLP + ReLU")

    summary = "\n".join(
        [
            "Day09 MLP / ReLU 非线性分类对比",
            "=" * 44,
            f"真实任务：XOR 风格二维二分类",
            f"Linear baseline final accuracy：{linear_acc:.3f}",
            f"MLP + ReLU final accuracy：{mlp_acc:.3f}",
            "",
            linear_log,
            "",
            linear_samples,
            "",
            mlp_log,
            "",
            mlp_samples,
            "",
            f"Linear 可视化：{linear_map_path}",
            f"MLP 可视化：{mlp_map_path}",
        ]
    )
    save_log(summary, log_path)

    print()
    print(f"Linear baseline final accuracy：{linear_acc:.3f}")
    print(f"MLP + ReLU final accuracy：{mlp_acc:.3f}")
    print(f"训练日志已保存到：{log_path}")
    print(f"Linear 分类图已保存到：{linear_map_path}")
    print(f"MLP 分类图已保存到：{mlp_map_path}")


if __name__ == "__main__":
    main()
