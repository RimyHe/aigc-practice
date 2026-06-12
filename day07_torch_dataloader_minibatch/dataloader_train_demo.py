from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_dataset(num_samples=100):
    torch.manual_seed(0)
    x = torch.linspace(-3, 3, num_samples).reshape(-1, 1)
    noise = torch.randn(num_samples, 1) * 0.3
    y = 3 * x + 2 + noise
    return x, y


def make_dataloader(x, y, batch_size=10, shuffle=True):
    dataset = TensorDataset(x, y)
    generator = torch.Generator().manual_seed(0)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )
    return dataset, dataloader


def get_weight_and_bias(model):
    return model.weight.item(), model.bias.item()


def inspect_first_batch(dataloader):
    batch_x, batch_y = next(iter(dataloader))
    lines = [
        "第一个 mini-batch：",
        f"- batch_x shape：{tuple(batch_x.shape)}",
        f"- batch_y shape：{tuple(batch_y.shape)}",
        f"- batch_x 前 3 个值：{batch_x[:3].reshape(-1).tolist()}",
        f"- batch_y 前 3 个值：{batch_y[:3].reshape(-1).tolist()}",
        "",
    ]
    return "\n".join(lines)


def train_model(dataloader, num_epochs=10, lr=0.1):
    model = nn.Linear(1, 1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    dataset_size = len(dataloader.dataset)

    log_lines = [
        "PyTorch DataLoader mini-batch 训练日志",
        "=" * 44,
        f"batch_size：{dataloader.batch_size}",
        f"每个 epoch 的 batch 数：{len(dataloader)}",
        f"学习率：{lr}",
        "",
        "epoch | avg_loss | weight | bias",
    ]

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0

        for batch_x, batch_y in dataloader:
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.shape[0]

        avg_loss = total_loss / dataset_size
        weight, bias = get_weight_and_bias(model)
        log_line = f"{epoch:>5} | {avg_loss:>8.5f} | {weight:>7.4f} | {bias:>7.4f}"
        log_lines.append(log_line)
        print(log_line)

    final_weight, final_bias = get_weight_and_bias(model)
    log_lines.extend(
        [
            "",
            f"训练后学到的 weight：{final_weight:.6f}",
            f"训练后学到的 bias：{final_bias:.6f}",
            "观察：weight 应该接近 3，bias 应该接近 2。",
        ]
    )
    return model, "\n".join(log_lines)


def save_log(log_text, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(log_text)


def main():
    project_dir = Path(__file__).parent
    output_path = project_dir / "outputs" / "training_log.txt"

    x, y = make_dataset(num_samples=100)
    dataset, dataloader = make_dataloader(x, y, batch_size=10, shuffle=True)

    print(f"样本总数：{len(dataset)}")
    first_batch_text = inspect_first_batch(dataloader)
    print(first_batch_text)

    model, log_text = train_model(dataloader, num_epochs=10, lr=0.1)
    weight, bias = get_weight_and_bias(model)
    full_log_text = first_batch_text + log_text
    save_log(full_log_text, output_path)

    print()
    print(f"训练后学到的 weight：{weight:.6f}")
    print(f"训练后学到的 bias：{bias:.6f}")
    print("真实参数：weight=3, bias=2")
    print(f"训练日志已保存到：{output_path}")


if __name__ == "__main__":
    main()
