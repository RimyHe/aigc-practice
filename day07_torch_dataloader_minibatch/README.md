# Day 07 - PyTorch DataLoader 和 Mini-batch

今天的目标是理解真实训练为什么常用 mini-batch，而不是一次把所有数据都塞进模型。你会把 Day06 的线性回归数据包装成 `TensorDataset`，再用 `DataLoader` 按 batch 喂给模型，练习 epoch/batch 双层训练循环。

## 今日目标

完成后你应该能说清楚：

1. `Dataset` 和 `DataLoader` 分别负责什么；
2. `batch_size` 如何改变每次喂给模型的数据形状；
3. `shuffle=True` 为什么通常用于训练集；
4. 一个 epoch 和一个 batch 有什么区别；
5. mini-batch 训练循环和 Day06 全量训练循环有什么不同。

## d2l PyTorch 对应参考

本练习对齐《动手学深度学习》PyTorch 版本：

- [3.2 线性回归的从零开始实现](https://zh.d2l.ai/chapter_linear-networks/linear-regression-scratch.html)：重点看小批量数据读取和 batch shape。
- [3.3 线性回归的简洁实现](https://zh.d2l.ai/chapter_linear-networks/linear-regression-concise.html)：重点看 PyTorch 的 `TensorDataset` / `DataLoader` 思想。
- [3.5 图像分类数据集](https://zh.d2l.ai/chapter_linear-networks/image-classification-dataset.html)：理解真实图像数据也通过 DataLoader 迭代读取。

注意：本练习只使用 PyTorch 写法，不使用 MXNet、TensorFlow 或 Paddle 写法。

## 文件结构

```text
day07_torch_dataloader_minibatch/
  README.md
  outputs/
    training_log.txt
  dataloader_train_demo.py
  notes.md
```

- `README.md`：今天的练习说明和逐行代码解释。
- `dataloader_train_demo.py`：主脚本，演示 `TensorDataset`、`DataLoader` 和 mini-batch 训练。
- `outputs/training_log.txt`：运行后生成的训练日志。
- `notes.md`：解释 mini-batch、DataLoader 和 AIGC 训练数据读取的关系。

## 先打开哪个文件

建议按这个顺序：

1. 先打开 `dataloader_train_demo.py`，重点看 `make_dataloader()` 和 `train_model()`。
2. 再运行脚本，看第一个 batch 的 shape。
3. 最后打开 `outputs/training_log.txt`，对照每个 epoch 的平均 loss、weight、bias。

可以用：

```powershell
cd D:\1AIGC_daily_practice\day07_torch_dataloader_minibatch
notepad dataloader_train_demo.py
```

## 安装依赖

今天需要 PyTorch。如果运行时报 `No module named 'torch'`，先安装：

```powershell
pip install torch
```

## 运行命令

进入当天文件夹：

```powershell
cd D:\1AIGC_daily_practice\day07_torch_dataloader_minibatch
```

运行脚本：

```powershell
python dataloader_train_demo.py
```

运行后打开日志：

```powershell
notepad outputs\training_log.txt
```

## 应该观察什么输出

你会先看到第一个 batch 的形状：

```text
样本总数：100
第一个 mini-batch：
- batch_x shape：(10, 1)
- batch_y shape：(10, 1)
```

这说明总共有 100 条样本，每次取 10 条，所以一个 batch 的输入是 `(10, 1)`。

然后会看到每个 epoch 的平均 loss、weight、bias。你要重点观察：

1. 平均 loss 是否整体下降；
2. weight 是否接近 3；
3. bias 是否接近 2；
4. 每个 epoch 内部其实遍历了多个 batch。

## 整体思路

Day06 是全量训练：每轮直接把全部 `x` 和 `y` 喂给模型。

Day07 改成 mini-batch 训练：

```text
生成数据 -> TensorDataset 包装样本 -> DataLoader 切 batch -> 双层循环训练 -> 保存日志
```

核心训练结构是：

```text
for epoch in epochs:
    for batch_x, batch_y in dataloader:
        pred = model(batch_x)
        loss = loss_fn(pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

外层 `epoch` 表示完整遍历数据集多少轮；内层 `batch` 表示每次从 DataLoader 取出一小批样本。

## 逐行解释代码

下面像老师带读一样，把 `dataloader_train_demo.py` 按代码块拆开。

### 1. 导入库

```python
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
```

逐行解释：

- `Path`：用来拼出 `outputs/training_log.txt`。
- `torch`：生成 Tensor、设置随机种子、创建随机噪声。
- `nn`：提供 `Linear` 和 `MSELoss`。
- `TensorDataset`：把 `x` 和 `y` 绑定成一个数据集。
- `DataLoader`：负责按 batch 读取数据，并可以 shuffle。

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

- `torch.manual_seed(0)`：固定随机种子，保证每次运行结果可复现。
- `torch.linspace(-3, 3, num_samples)`：生成 100 个 x。
- `.reshape(-1, 1)`：把 x 整理成 `(100, 1)`，这是模型需要的二维形状。
- `noise`：加入噪声，让数据更接近真实实验。
- `y = 3 * x + 2 + noise`：真实规律，训练后模型应该学到接近 3 和 2 的参数。

### 3. 包装 Dataset 和 DataLoader

```python
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
```

逐行解释：

- `TensorDataset(x, y)`：把输入和标签按样本维度绑在一起，保证第 i 个 x 对应第 i 个 y。
- `batch_size=10`：每次取 10 条样本。
- `shuffle=True`：每个 epoch 打乱样本顺序，避免模型总是按固定顺序看数据。
- `generator`：固定 DataLoader 打乱顺序的随机性，方便复现实验。
- `return dataset, dataloader`：返回原始数据集和批量迭代器。

### 4. 查看第一个 batch

```python
def inspect_first_batch(dataloader):
    batch_x, batch_y = next(iter(dataloader))
```

逐行解释：

- `iter(dataloader)`：把 DataLoader 变成一个可以逐批取数据的迭代器。
- `next(...)`：取出第一个 batch。
- `batch_x, batch_y`：分别是这一批的输入和标签。

如果 `batch_size=10`，那么：

```text
batch_x shape = (10, 1)
batch_y shape = (10, 1)
```

### 5. 定义模型、loss 和 optimizer

```python
model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
```

逐行解释：

- `nn.Linear(1, 1)`：最小线性模型，内部有 weight 和 bias。
- `MSELoss`：衡量预测和真实 y 的平均平方误差。
- `model.parameters()`：把模型参数交给优化器管理。
- `SGD`：根据梯度更新参数。

### 6. 双层训练循环

```python
for epoch in range(1, num_epochs + 1):
    total_loss = 0.0

    for batch_x, batch_y in dataloader:
        pred = model(batch_x)
        loss = loss_fn(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

逐行解释：

- 外层 `for epoch ...`：完整遍历数据集多轮。
- `total_loss = 0.0`：记录当前 epoch 的总损失。
- 内层 `for batch_x, batch_y in dataloader`：每次取一个 mini-batch。
- `pred = model(batch_x)`：只对当前 batch 做 forward。
- `loss = loss_fn(pred, batch_y)`：只计算当前 batch 的 loss。
- `optimizer.zero_grad()`：清空上一个 batch 留下的梯度。
- `loss.backward()`：计算当前 batch 的梯度。
- `optimizer.step()`：用当前 batch 的梯度更新参数。

这就是今天最重要的变化：训练循环从一层变成两层。

### 7. 计算 epoch 平均 loss

```python
total_loss += loss.item() * batch_x.shape[0]
avg_loss = total_loss / dataset_size
```

逐行解释：

- `loss.item()` 是当前 batch 的平均 loss。
- `batch_x.shape[0]` 是当前 batch 有多少样本。
- 乘起来后加到 `total_loss`，表示按样本数累计损失。
- 最后除以 `dataset_size`，得到整个 epoch 的平均 loss。

这样写比直接平均每个 batch 的 loss 更稳，因为最后一个 batch 可能不满。

### 8. 保存和打印结果

```python
weight, bias = get_weight_and_bias(model)
log_line = f"{epoch:>5} | {avg_loss:>8.5f} | {weight:>7.4f} | {bias:>7.4f}"
log_lines.append(log_line)
print(log_line)
```

逐行解释：

- `get_weight_and_bias(model)`：读取当前模型参数。
- `log_line`：把 epoch、平均 loss、weight、bias 格式化成一行。
- `log_lines.append(...)`：保存到日志文件。
- `print(...)`：同时打印到终端，方便你直接观察训练过程。

## 重点理解的 5 个代码点

### 1. `TensorDataset`

```python
dataset = TensorDataset(x, y)
```

它把输入和标签绑定成样本集合，保证每个 `x` 和对应的 `y` 不会错位。

### 2. `DataLoader`

```python
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
```

它把 Dataset 变成可以按 batch 迭代的数据读取器。

### 3. `batch_size`

`batch_size=10` 表示每次取 10 条样本。总样本数 100 时，每个 epoch 有 10 个 batch。

### 4. `shuffle=True`

训练时通常打乱样本顺序，减少固定顺序带来的偏差。

### 5. epoch / batch 双层循环

```text
epoch：完整看完整个数据集一遍
batch：一次取出的一小批样本
```

真实训练代码几乎都离不开这两层。

## 常见错误

### 1. `x/y` 样本数不一致

`TensorDataset(x, y)` 要求第一维样本数一致。比如 `x` 是 100 条，`y` 也必须是 100 条。

### 2. batch shape 看不懂

`(10, 1)` 不是 10 行文本，而是 10 条样本，每条样本 1 个特征。

### 3. 忘记遍历 DataLoader

不要把整个 DataLoader 直接丢给模型。正确做法是：

```python
for batch_x, batch_y in dataloader:
    pred = model(batch_x)
```

### 4. 把 epoch 和 batch 混淆

epoch 是完整遍历数据集一轮；batch 是这一轮里的某一小批。

### 5. loss 平均方式错误

如果最后一个 batch 不满，直接平均 batch loss 可能有偏差。更稳的方式是按样本数加权：

```python
total_loss += loss.item() * batch_x.shape[0]
avg_loss = total_loss / dataset_size
```

## 自查清单和参考答案

- 问：100 个样本、`batch_size=10` 时，每个 epoch 有几个 batch？
  答：10 个。

- 问：为什么 `batch_x shape` 是 `(10, 1)`？
  答：10 是 batch 里的样本数，1 是每个样本的特征数。

- 问：`shuffle=True` 改变了什么？
  答：它会打乱每个 epoch 中样本被读取的顺序。

- 问：mini-batch 训练和 Day06 全量训练最大的区别是什么？
  答：Day06 每轮用全部数据算一次 loss；Day07 每轮内部要遍历多个 batch，每个 batch 都会更新一次参数。

## 可修改的小练习

### 练习 1：修改 batch_size

把：

```python
batch_size=10
```

改成：

```python
batch_size=20
```

参考结果：每个 epoch 的 batch 数会从 10 变成 5，`batch_x shape` 会变成 `(20, 1)`。

### 练习 2：关掉 shuffle

把：

```python
shuffle=True
```

改成：

```python
shuffle=False
```

参考结果：第一个 batch 会固定从最前面的 x 开始取。训练仍能收敛，但真实训练通常会打开 shuffle。

### 练习 3：把样本数从 100 改到 1000

把：

```python
make_dataset(num_samples=100)
```

改成：

```python
make_dataset(num_samples=1000)
```

参考结果：`batch_size=10` 时，每个 epoch 会有 100 个 batch，训练日志里的平均 loss 会更稳定。

## 面试表达

可以这样总结今天的练习：

这个练习用 `TensorDataset` 和 `DataLoader` 把线性回归数据改造成 mini-batch 训练流程。我理解了 Dataset 负责组织样本，DataLoader 负责按 batch 迭代数据，而训练循环从单层 epoch 变成 epoch/batch 双层结构。这个模式可以迁移到图像分类、CLIP 图文对训练、VAE 重建、Stable Diffusion 和 DiT 的训练数据读取。
