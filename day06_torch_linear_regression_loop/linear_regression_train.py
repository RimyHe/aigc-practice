from pathlib import Path

import torch
from torch import nn


def make_dataset(num_samples=100, a=3, b=2):
    torch.manual_seed(0)
    x = torch.linspace(-3, 3, num_samples).reshape(-1, 1)
    noise = torch.randn(num_samples, 1) * 0.3
    y = a * x + b + noise
    return x, y, a, b


def get_weight_and_bias(model):
    weight = model.weight.item()
    bias = model.bias.item()
    return weight, bias


def train_model(x, y, num_epochs=50, lr=0.1, true_a=3, true_b=2):
    model = nn.Linear(1, 1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    log_lines = [
        "PyTorch 线性回归训练日志",
        "=" * 32,
        f"真实函数：y = {true_a} * x + {true_b} + noise",
        f"样本数量：{len(x)}",
        f"学习率：{lr}",
        "",
        "epoch | loss | weight | bias",
    ]

    print_epochs = {1, 2, 3, 5, 10, 20, 30, 40, num_epochs}

    for epoch in range(1, num_epochs + 1):
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        weight, bias = get_weight_and_bias(model)
        log_line = f"{epoch:>5} | {loss.item():>8.5f} | {weight:>7.4f} | {bias:>7.4f}"
        log_lines.append(log_line)

        if epoch in print_epochs:
            print(log_line)

    final_weight, final_bias = get_weight_and_bias(model)
    log_lines.extend(
        [
            "",
            f"训练后学到的 weight：{final_weight:.6f}",
            f"训练后学到的 bias：{final_bias:.6f}",
            f"观察：weight 应该接近 {true_a}，bias 应该接近 {true_b}。",
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

    x, y, a, b = make_dataset(num_samples=100, a=5, b=-1)
    model, log_text = train_model(x, y, num_epochs=80, lr=0.08, true_a=a, true_b=b)
    weight, bias = get_weight_and_bias(model)
    save_log(log_text, output_path)

    print()
    print(f"训练后学到的 weight：{weight:.6f}")
    print(f"训练后学到的 bias：{bias:.6f}")
    print(f"真实参数：weight={a}, bias={b}")
    print(f"训练日志已保存到：{output_path}")


if __name__ == "__main__":
    main()
