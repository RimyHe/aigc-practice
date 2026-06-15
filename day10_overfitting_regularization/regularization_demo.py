from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_toy_data(num_samples, seed, label_noise=0.0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand((num_samples, 2), generator=generator) * 6.0 - 3.0

    score = x[:, 0] * x[:, 1] + 0.6 * torch.sin(1.5 * x[:, 0])
    y = (score > 0).long()

    if label_noise > 0:
        flip_mask = torch.rand(num_samples, generator=generator) < label_noise
        y[flip_mask] = 1 - y[flip_mask]

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


def build_big_mlp(dropout_p=0.0):
    layers = [
        nn.Linear(2, 128),
        nn.ReLU(),
    ]
    if dropout_p > 0:
        layers.append(nn.Dropout(p=dropout_p))

    layers.extend(
        [
            nn.Linear(128, 128),
            nn.ReLU(),
        ]
    )
    if dropout_p > 0:
        layers.append(nn.Dropout(p=dropout_p))

    layers.append(nn.Linear(128, 2))
    return nn.Sequential(*layers)


def evaluate(model, x, y, loss_fn):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = loss_fn(logits, y).item()
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean().item()
    return loss, acc


def train_one_model(name, train_x, train_y, test_x, test_y, weight_decay=0.0, dropout_p=0.0):
    torch.manual_seed(0)
    model = build_big_mlp(dropout_p=dropout_p)
    loss_fn = nn.CrossEntropyLoss()
    _, train_loader = make_dataloader(train_x, train_y, batch_size=16)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        weight_decay=weight_decay,
    )

    lines = [
        f"{name} 训练日志",
        "=" * 40,
        f"weight_decay: {weight_decay}",
        f"dropout_p: {dropout_p}",
        "epoch | train_loss | train_acc | test_acc | gap",
    ]

    for epoch in range(1, 101):
        model.train()
        for batch_x, batch_y in train_loader:
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss, train_acc = evaluate(model, train_x, train_y, loss_fn)
        test_loss, test_acc = evaluate(model, test_x, test_y, loss_fn)
        gap = train_acc - test_acc
        line = f"{epoch:>5} | {train_loss:>10.5f} | {train_acc:>9.3f} | {test_acc:>8.3f} | {gap:>5.3f}"
        lines.append(line)

        if epoch in {1, 2, 3, 5, 10, 20, 50, 100}:
            print(f"{name:>14} {line}")

    return {
        "name": name,
        "model": model,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "gap": gap,
        "log": "\n".join(lines),
    }


def format_summary(results):
    lines = [
        "Day10 过拟合与正则化实验汇总",
        "=" * 48,
        "数据设置：训练集 40 个点，其中 15% 标签被故意翻转；测试集 400 个点，不加标签噪声。",
        "观察重点：不要只看 train_acc，要看 test_acc 以及 train_acc - test_acc 的 gap。",
        "",
        "model | train_loss | test_loss | train_acc | test_acc | gap",
    ]

    for result in results:
        lines.append(
            f"{result['name']} | "
            f"{result['train_loss']:.5f} | "
            f"{result['test_loss']:.5f} | "
            f"{result['train_acc']:.3f} | "
            f"{result['test_acc']:.3f} | "
            f"{result['gap']:.3f}"
        )

    best_train = max(results, key=lambda item: item["train_acc"])
    best_test = max(results, key=lambda item: item["test_acc"])
    smallest_gap = min(results, key=lambda item: item["gap"])

    lines.extend(
        [
            "",
            f"训练集 accuracy 最高：{best_train['name']} ({best_train['train_acc']:.3f})",
            f"测试集 accuracy 最高：{best_test['name']} ({best_test['test_acc']:.3f})",
            f"train/test gap 最小：{smallest_gap['name']} ({smallest_gap['gap']:.3f})",
            "",
            "结论：baseline 最容易记住小训练集；weight decay 和 dropout 会牺牲一点训练集表现，换取更稳的泛化表现。",
        ]
    )

    for result in results:
        lines.extend(["", result["log"]])

    return "\n".join(lines)


def main():
    project_dir = Path(__file__).parent
    output_dir = project_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "regularization_summary.txt"

    train_x, train_y = make_toy_data(num_samples=40, seed=0, label_noise=0.15)
    test_x, test_y = make_toy_data(num_samples=400, seed=1, label_noise=0.0)
    print(f"训练集 x shape：{tuple(train_x.shape)}")
    print(f"训练集 y shape：{tuple(train_y.shape)}")
    print(f"测试集 x shape：{tuple(test_x.shape)}")
    print(f"测试集 y shape：{tuple(test_y.shape)}")
    print()

    configs = [
        {"name": "baseline", "weight_decay": 0.0, "dropout_p": 0.0},
        {"name": "weight_decay", "weight_decay": 3e-3, "dropout_p": 0.0},
        {"name": "dropout", "weight_decay": 0.0, "dropout_p": 0.3},
    ]

    results = []
    for config in configs:
        result = train_one_model(
            name=config["name"],
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y,
            weight_decay=config["weight_decay"],
            dropout_p=config["dropout_p"],
        )
        results.append(result)
        print()

    summary = format_summary(results)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print("最终对比：")
    print("model | train_acc | test_acc | gap")
    for result in results:
        print(f"{result['name']} | {result['train_acc']:.3f} | {result['test_acc']:.3f} | {result['gap']:.3f}")

    best_test = max(results, key=lambda item: item["test_acc"])
    best_train = max(results, key=lambda item: item["train_acc"])
    print()
    print(f"训练集最高的是：{best_train['name']}")
    print(f"测试集更稳的是：{best_test['name']}")
    print(f"实验汇总已保存到：{summary_path}")


if __name__ == "__main__":
    main()
