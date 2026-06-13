from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image, ImageDraw


CLASS_NAMES = ["red", "green", "blue"]
CLASS_COLORS = [
    (230, 80, 80),
    (80, 180, 100),
    (80, 120, 230),
]


def make_dataset(samples_per_class=20):
    torch.manual_seed(0)
    centers = torch.tensor(
        [
            [-2.0, -2.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ]
    )

    features = []
    labels = []
    for class_id, center in enumerate(centers):
        points = center + torch.randn(samples_per_class, 2) * 0.45
        class_labels = torch.full((samples_per_class,), class_id, dtype=torch.long)
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


def accuracy_from_logits(logits, labels):
    predicted = logits.argmax(dim=1)
    correct = (predicted == labels).sum().item()
    return correct / labels.numel()


def describe_samples(model, x, y, title, sample_indices=None):
    if sample_indices is None:
        sample_indices = [0, 20, 40, 5, 25]

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


def train_model(model, dataloader, num_epochs=50, lr=0.05):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    dataset_size = len(dataloader.dataset)
    log_lines = [
        "PyTorch softmax / cross-entropy 训练日志",
        "=" * 46,
        f"类别数：{len(CLASS_NAMES)}",
        f"batch_size：{dataloader.batch_size}",
        f"每个 epoch 的 batch 数：{len(dataloader)}",
        f"学习率：{lr}",
        "",
        "epoch | avg_loss | accuracy",
    ]

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        total_correct = 0

        model.train()
        for batch_x, batch_y in dataloader:
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.shape[0]
            total_correct += (logits.argmax(dim=1) == batch_y).sum().item()

        avg_loss = total_loss / dataset_size
        accuracy = total_correct / dataset_size
        log_line = f"{epoch:>5} | {avg_loss:>8.5f} | {accuracy:>8.3f}"
        log_lines.append(log_line)

        if epoch in {1, 2, 3, 5, 10, 20, 30, 40, num_epochs}:
            print(log_line)

    return "\n".join(log_lines)


def save_classification_map(model, x, y, output_path):
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
    zero_x, _ = to_pixel([0.0, 0.0])
    _, zero_y = to_pixel([0.0, 0.0])
    draw.line((zero_x, 0, zero_x, height), fill=axis_color, width=1)
    draw.line((0, zero_y, width, zero_y), fill=axis_color, width=1)

    legend_x = 12
    legend_y = 12
    for i, name in enumerate(CLASS_NAMES):
        y0 = legend_y + i * 22
        draw.rectangle((legend_x, y0, legend_x + 14, y0 + 14), fill=CLASS_COLORS[i])
        draw.text((legend_x + 20, y0 - 2), name, fill=(0, 0, 0))

    image.save(output_path)


def save_log(log_text, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(log_text)


def main():
    project_dir = Path(__file__).parent
    output_path = project_dir / "outputs" / "training_log.txt"
    map_path = project_dir / "outputs" / "classification_map.png"

    x, y = make_dataset(samples_per_class=20)
    dataset, dataloader = make_dataloader(x, y, batch_size=16)
    model = nn.Linear(2, 3)

    print(f"样本总数：{len(dataset)}")
    print(f"x shape：{tuple(x.shape)}")
    print(f"y shape：{tuple(y.shape)}")
    print()

    before_text = describe_samples(model, x, y, "训练前样本预测：")
    print(before_text)
    print()

    log_text = train_model(model, dataloader, num_epochs=50, lr=0.2)
    after_text = describe_samples(model, x, y, "训练后样本预测：")
    full_log_text = before_text + "\n\n" + log_text + "\n\n" + after_text
    save_log(full_log_text, output_path)
    save_classification_map(model, x, y, map_path)

    print()
    print(after_text)
    print(f"训练日志已保存到：{output_path}")
    print(f"分类可视化已保存到：{map_path}")


if __name__ == "__main__":
    main()
