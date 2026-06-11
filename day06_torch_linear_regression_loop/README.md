# Day 06 - PyTorch 线性回归训练循环

今天的目标是从 Day05 的单参数 `w * x` autograd，过渡到一个完整但很小的 PyTorch 训练流程。你会训练一个 `torch.nn.Linear(1, 1)`，让它从带噪声的数据里学到接近 `weight=3`、`bias=2` 的线性关系。

## 今日目标

完成后你应该能说清楚：

1. 线性回归为什么是深度学习训练循环的最小完整模型；
2. `nn.Linear(1, 1)` 里自动管理了哪些参数；
3. `MSELoss` 如何衡量预测值和真实值的差距；
4. `SGD` 如何替代 Day05 里手写的 `w -= lr * grad`；
5. 标准训练循环为什么是 `pred -> loss -> zero_grad -> backward -> step`。

## d2l PyTorch 对应参考

本练习对齐《动手学深度学习》PyTorch 版本：

- [3.1 线性回归](https://zh.d2l.ai/chapter_linear-networks/linear-regression.html)：理解线性关系、特征、标签、噪声和损失函数。
- [3.2 线性回归的从零开始实现](https://zh.d2l.ai/chapter_linear-networks/linear-regression-scratch.html)：理解合成数据、手动 SGD、训练循环。
- [3.3 线性回归的简洁实现](https://zh.d2l.ai/chapter_linear-networks/linear-regression-concise.html)：对应本练习里的 `nn.Linear`、`nn.MSELoss`、`torch.optim.SGD`。

注意：本练习只使用 PyTorch 写法，不使用 MXNet、TensorFlow 或 Paddle 写法。

## 文件结构

```text
day06_torch_linear_regression_loop/
  README.md
  outputs/
    training_log.txt
  linear_regression_train.py
  notes.md
```

- `README.md`：今天的练习说明和逐行代码解释。
- `linear_regression_train.py`：主脚本，生成数据并训练线性回归模型。
- `outputs/training_log.txt`：运行后生成的训练日志。
- `notes.md`：解释线性回归、MSE、optimizer 和 AIGC 训练流程的关系。

## 先打开哪个文件

建议按这个顺序：

1. 先打开 `linear_regression_train.py`，看 `main()` 调用了哪些函数。
2. 再运行脚本，看 loss、weight、bias 如何变化。
3. 最后打开 `outputs/training_log.txt`，对照每一轮训练记录。

可以用：

```powershell
cd D:\1AIGC_daily_practice\day06_torch_linear_regression_loop
notepad linear_regression_train.py
```

## 安装依赖

今天需要 PyTorch。如果运行时报 `No module named 'torch'`，先安装：

```powershell
pip install torch
```

如果安装慢或失败，把报错贴出来，再按你的环境处理。

## 运行命令

进入当天文件夹：

```powershell
cd D:\1AIGC_daily_practice\day06_torch_linear_regression_loop
```

运行脚本：

```powershell
python linear_regression_train.py
```

运行后打开日志：

```powershell
notepad outputs\training_log.txt
```

## 应该观察什么输出

终端会打印若干轮训练记录，类似：

```text
epoch | loss | weight | bias
    1 | 36.26249 |  1.7397 |  0.4383
    2 |  7.42816 |  2.5111 |  0.7529
   10 |  0.16408 |  2.9996 |  1.8000
   50 |  0.09445 |  2.9999 |  2.0111

训练后学到的 weight：2.999866
训练后学到的 bias：2.011114
真实参数：weight=3, bias=2
```

你要重点观察：

1. loss 是否逐步下降；
2. weight 是否逐步接近 3；
3. bias 是否逐步接近 2；
4. 日志文件是否保存到了 `outputs/training_log.txt`。

## 整体思路

这个练习模拟的是最小监督学习训练流程：

```text
生成数据 -> 定义模型 -> 定义 loss -> 定义 optimizer -> 训练多轮 -> 保存日志
```

具体步骤：

1. 生成 100 个样本，真实关系是：

```text
y = 3 * x + 2 + noise
```

2. 用 `nn.Linear(1, 1)` 定义模型。它内部有一个 weight 和一个 bias。

3. 用 `MSELoss` 衡量预测值和真实 `y` 的差距。

4. 用 `SGD` 根据梯度更新模型参数。

5. 每个 epoch 执行：

```text
pred = model(x)
loss = loss_fn(pred, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

6. 保存每轮的 epoch、loss、weight、bias。

## 逐行解释代码

下面像老师带读一样，把 `linear_regression_train.py` 按代码块拆开。

### 1. 导入库

```python
from pathlib import Path

import torch
from torch import nn
```

逐行解释：

- `from pathlib import Path`：用来稳定拼出 `outputs/training_log.txt` 的路径。
- `import torch`：导入 PyTorch 主库，用来生成 Tensor、固定随机种子、创建优化器。
- `from torch import nn`：导入神经网络模块。`nn.Linear` 和 `nn.MSELoss` 都在这里。

工程联系：真实训练脚本通常会同时导入路径工具、PyTorch、模型模块和数据处理模块。

### 2. 生成模拟数据

```python
def make_dataset(num_samples=100):
    torch.manual_seed(0)
    x = torch.linspace(-3, 3, num_samples).reshape(-1, 1)
    noise = torch.randn(num_samples, 1) * 0.3
    y = 3 * x + 2 + noise
    return x, y
```

逐行解释：

- `def make_dataset(num_samples=100):`：定义生成数据的函数，默认生成 100 条样本。
- `torch.manual_seed(0)`：固定随机种子。这样每次运行得到的噪声一致，便于复现实验。
- `torch.linspace(-3, 3, num_samples)`：在 -3 到 3 之间均匀生成 100 个 x。
- `.reshape(-1, 1)`：把 x 变成 `(100, 1)`，也就是 100 行、1 个特征。`nn.Linear(1, 1)` 需要这种二维输入。
- `torch.randn(num_samples, 1) * 0.3`：生成噪声。噪声让数据不完全落在直线上，更像真实实验。
- `y = 3 * x + 2 + noise`：真实规律。模型训练后应该学到接近 3 的 weight 和接近 2 的 bias。
- `return x, y`：返回输入和标签。

### 3. 读取模型参数

```python
def get_weight_and_bias(model):
    weight = model.weight.item()
    bias = model.bias.item()
    return weight, bias
```

逐行解释：

- `model.weight`：`nn.Linear(1, 1)` 里的权重参数。
- `.item()`：因为这里只是一个数字，所以取出 Python 数值，方便打印日志。
- `model.bias`：线性模型里的偏置参数。
- `return weight, bias`：返回当前模型学到的两个参数。

注意：`.item()` 适合记录数值，不要在需要继续反向传播的计算里乱用。

### 4. 定义模型、loss 和 optimizer

```python
model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
```

逐行解释：

- `nn.Linear(1, 1)`：定义一个线性层。输入维度是 1，输出维度是 1。它内部自动创建 weight 和 bias。
- `nn.MSELoss()`：均方误差损失。预测越接近真实 `y`，loss 越小。
- `model.parameters()`：把模型里的所有可训练参数交给 optimizer。
- `torch.optim.SGD(..., lr=lr)`：使用随机梯度下降优化器，根据梯度更新参数。

工程联系：在 CLIP、VAE、Stable Diffusion、DiT 里，模型比 `nn.Linear` 大很多，但也都是把 `model.parameters()` 交给 optimizer 管理。

### 5. 训练循环

```python
for epoch in range(1, num_epochs + 1):
    pred = model(x)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

逐行解释：

- `for epoch in range(1, num_epochs + 1):`：训练多轮。每一轮都用当前参数重新预测、算 loss、更新参数。
- `pred = model(x)`：前向传播。模型根据当前 weight 和 bias 对所有 x 做预测。
- `loss = loss_fn(pred, y)`：计算预测和真实标签的均方误差。
- `optimizer.zero_grad()`：清空上一轮残留的梯度。
- `loss.backward()`：反向传播，自动计算 weight 和 bias 的梯度。
- `optimizer.step()`：根据梯度更新 weight 和 bias。

这是今天最重要的代码块。以后看任何 PyTorch 训练代码，都先找这五步。

### 6. 记录训练日志

```python
weight, bias = get_weight_and_bias(model)
log_line = f"{epoch:>5} | {loss.item():>8.5f} | {weight:>7.4f} | {bias:>7.4f}"
log_lines.append(log_line)
```

逐行解释：

- `get_weight_and_bias(model)`：取出当前模型参数。
- `loss.item()`：把 loss 这个单值 Tensor 转成 Python 数字。
- `f"{...}"`：把 epoch、loss、weight、bias 格式化成表格。
- `log_lines.append(log_line)`：保存到日志列表，最后写入文件。

工程联系：实验日志能帮你判断模型是否真的在学习。没有日志，就很难复盘训练是否成功。

### 7. 保存日志

```python
def save_log(log_text, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(log_text)
```

逐行解释：

- `output_path.parent.mkdir(...)`：如果 `outputs/` 不存在，就自动创建。
- `open(..., "w", encoding="utf-8")`：用 UTF-8 写入日志，避免中文乱码。
- `f.write(log_text)`：保存完整训练记录。

### 8. 主入口

```python
def main():
    project_dir = Path(__file__).parent
    output_path = project_dir / "outputs" / "training_log.txt"

    x, y = make_dataset(num_samples=100)
    model, log_text = train_model(x, y, num_epochs=50, lr=0.1)
    weight, bias = get_weight_and_bias(model)
    save_log(log_text, output_path)
```

逐行解释：

- `project_dir = Path(__file__).parent`：找到当前脚本所在目录。
- `output_path = ...`：拼出日志保存位置。
- `make_dataset(...)`：生成训练数据。
- `train_model(...)`：训练线性模型。
- `get_weight_and_bias(model)`：取出最终学到的参数。
- `save_log(...)`：保存训练日志。

整体流程就是：

```text
定位路径 -> 生成数据 -> 训练模型 -> 读取参数 -> 保存日志 -> 打印结果
```

### 9. 入口保护

```python
if __name__ == "__main__":
    main()
```

逐行解释：

- 如果直接运行这个文件，就执行 `main()`。
- 如果以后被别的脚本导入，不会自动开始训练。

## 重点理解的 5 个代码点

### 1. `nn.Linear`

```python
model = nn.Linear(1, 1)
```

它表示一个最小线性模型：

```text
pred = weight * x + bias
```

`nn.Linear` 会自动创建并管理 weight 和 bias。

### 2. `MSELoss`

```python
loss_fn = nn.MSELoss()
```

它衡量预测值和真实值之间的平均平方误差。预测越准，loss 越小。

### 3. `SGD`

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
```

它负责根据梯度更新模型参数，相当于替代 Day05 里的手写 `w -= lr * w.grad`。

### 4. `optimizer.zero_grad()`

```python
optimizer.zero_grad()
```

PyTorch 默认会累积梯度。每一轮 backward 前先清空旧梯度，避免上一轮的梯度影响这一轮。

### 5. `optimizer.step()`

```python
optimizer.step()
```

根据当前梯度更新模型参数。执行完这一步，weight 和 bias 才真正发生变化。

## 常见错误

### 1. 忘记 `zero_grad`

如果忘记：

```python
optimizer.zero_grad()
```

梯度会累积，loss 和参数更新可能变得异常。

### 2. `x/y` shape 不匹配

本练习里 `x` 和 `y` 都是 `(100, 1)`。如果 `x` 是 `(100,)`，有时会造成和模型输出 shape 不一致。

参考答案：用 `.reshape(-1, 1)` 把一维数据整理成二维列向量。

### 3. 学习率过大导致 loss 不降

如果把 `lr=0.1` 改得太大，比如 `lr=10`，参数可能来回震荡，loss 不降反升。

### 4. 把模型参数和普通 Tensor 混淆

`model.weight` 和 `model.bias` 是被 `nn.Linear` 管理的参数，不是随便创建的普通 Tensor。optimizer 会通过 `model.parameters()` 找到它们并更新。

## 自查清单和参考答案

- 问：为什么最终 weight 应该接近 3？
  答：因为数据真实关系是 `y = 3 * x + 2 + noise`。

- 问：为什么最终 bias 应该接近 2？
  答：因为真实函数里常数项是 2。

- 问：为什么训练结果不是精确等于 3 和 2？
  答：因为数据里加入了噪声，而且训练是用有限样本和优化过程估计参数。

- 问：`optimizer.step()` 做了什么？
  答：它根据 `loss.backward()` 算出的梯度更新模型参数。

- 问：这个例子和真实深度学习训练有什么关系？
  答：真实模型更复杂，但核心顺序仍然是 forward、loss、zero_grad、backward、step。

## 可修改的小练习

### 练习 1：把真实函数改成 `5 * x - 1`

指引：在 `make_dataset()` 里把：

```python
y = 3 * x + 2 + noise
```

改成：

```python
y = 5 * x - 1 + noise
```

参考结果：训练后的 weight 应该接近 5，bias 应该接近 -1。

### 练习 2：修改学习率

指引：在 `main()` 里把：

```python
train_model(x, y, num_epochs=50, lr=0.1)
```

改成：

```python
train_model(x, y, num_epochs=50, lr=0.01)
```

参考结果：loss 仍然会下降，但收敛速度变慢，50 轮后可能还没那么接近真实参数。

### 练习 3：增加 epoch

指引：把：

```python
num_epochs=50
```

改成：

```python
num_epochs=100
```

参考结果：如果学习率合适，参数会更稳定地接近真实值，loss 变化也会更平滑。

## 面试表达

可以这样总结今天的练习：

这个练习用 PyTorch 的 `nn.Linear`、`MSELoss` 和 `SGD` 跑通了最小监督学习训练循环。我生成了带噪声的线性数据，通过 forward 得到预测，计算 MSE loss，清空梯度后反向传播，再用 optimizer 更新参数。这个流程就是更复杂模型训练的基础骨架。
