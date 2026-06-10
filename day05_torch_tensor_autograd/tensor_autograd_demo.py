from pathlib import Path

import torch


def format_tensor_value(value):
    return f"{value.item():.6f}"


def run_one_step_demo(lines):
    lines.append("一、单步 autograd 演示")
    lines.append("-" * 32)

    normal_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    lines.append(f"普通 tensor：\n{normal_tensor}")
    lines.append(f"shape：{tuple(normal_tensor.shape)}")
    lines.append(f"dtype：{normal_tensor.dtype}")
    lines.append("")

    x = torch.tensor(2.0)
    y = torch.tensor(10.0)
    w = torch.tensor(1.0, requires_grad=True)
    lr = 0.1

    pred = w * x
    loss = (pred - y) ** 2
    loss.backward()

    lines.append(f"初始参数 w：{format_tensor_value(w)}")
    lines.append(f"x：{format_tensor_value(x)}")
    lines.append(f"目标 y：{format_tensor_value(y)}")
    lines.append(f"预测 pred = w * x：{format_tensor_value(pred)}")
    lines.append(f"loss = (pred - y) ** 2：{format_tensor_value(loss)}")
    lines.append(f"w.grad：{format_tensor_value(w.grad)}")
    lines.append("解释：w.grad 表示 loss 对 w 的导数，也就是 w 往哪个方向改会影响 loss。")

    with torch.no_grad():
        w -= lr * w.grad

    lines.append(f"手动更新后 w：{format_tensor_value(w)}")
    w.grad.zero_()
    lines.append(f"清空梯度后 w.grad：{format_tensor_value(w.grad)}")
    lines.append("")


def run_training_loop(lines):
    lines.append("二、10 步最小优化过程")
    lines.append("-" * 32)

    x = torch.tensor(3.0)
    y = torch.tensor(12.0)
    w = torch.tensor(1.0, requires_grad=True)
    lr = 0.08

    lines.append(
        f"目标：让 w * x 接近 y。这里 x={format_tensor_value(x)}，"
        f"y={format_tensor_value(y)}，所以理想 w 接近 {format_tensor_value(y / x)}。"
    )
    lines.append("step | w_before | loss | grad | w_after")

    for step in range(1, 31):
        pred = w * x
        loss = (pred - y) ** 2
        loss.backward()

        w_before = w.item()
        grad = w.grad.item()
        loss_value = loss.item()

        with torch.no_grad():
            w -= lr * w.grad

        w_after = w.item()
        lines.append(
            f"{step:>4} | {w_before:>8.4f} | {loss_value:>8.4f} | "
            f"{grad:>8.4f} | {w_after:>8.4f}"
        )

        w.grad.zero_()

    lines.append("")
    lines.append(f"训练结束后 w：{format_tensor_value(w)}")
    lines.append(f"此时 w * x：{format_tensor_value(w * x)}")
    lines.append(
        f"观察：loss 逐步下降，w 从 1.0 朝 {format_tensor_value(y / x)} 附近变化。"
    )


def save_log(log_text, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(log_text)


def main():
    project_dir = Path(__file__).parent
    output_path = project_dir / "outputs" / "training_log.txt"

    lines = [
        "PyTorch Tensor 与 Autograd 最小演示",
        "=" * 40,
        "",
    ]

    run_one_step_demo(lines)
    run_training_loop(lines)

    log_text = "\n".join(lines)
    save_log(log_text, output_path)

    print(log_text)
    print()
    print(f"训练日志已保存到：{output_path}")


if __name__ == "__main__":
    main()
