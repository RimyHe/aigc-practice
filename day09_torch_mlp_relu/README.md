# Day 09 - PyTorch MLP、ReLU 和非线性分类

今天的目标是理解为什么线性模型不够，以及 `Linear -> ReLU -> Linear` 的 MLP 如何给模型加入非线性表达能力。你会用一个 XOR 风格二维二分类数据集，对比线性 baseline 和 MLP + ReLU 的分类效果。

## 今日目标

完成后你应该能说清楚：

1. 为什么 XOR 风格数据不能被简单线性模型很好分开；
2. `nn.Sequential` 如何把多个层串成一个模型；
3. `nn.ReLU()` 为什么能引入非线性；
4. 隐藏层维度 `16` 表示什么；
5. `Linear -> ReLU -> Linear` 如何迁移到 CLIP projector、Transformer/DiT FFN。

## d2l PyTorch 对应参考

本练习对齐《动手学深度学习》PyTorch 版本：

- [4.1 多层感知机](https://zh.d2l.ai/chapter_multilayer-perceptrons/mlp.html)：理解隐藏层、激活函数和非线性。
- [4.2 多层感知机的从零开始实现](https://zh.d2l.ai/chapter_multilayer-perceptrons/mlp-scratch.html)：理解 MLP 的基本训练流程。
- [4.3 多层感知机的简洁实现](https://zh.d2l.ai/chapter_multilayer-perceptrons/mlp-concise.html)：对应本练习里的 `nn.Sequential`、`nn.Linear`、`nn.ReLU`。

注意：本练习只使用 PyTorch 写法，不使用 MXNet、TensorFlow 或 Paddle 写法。

## 文件结构

```text
day09_torch_mlp_relu/
  README.md
  outputs/
    training_log.txt
    linear_boundary.png
    mlp_boundary.png
  mlp_relu_demo.py
  notes.md
```

- `README.md`：今天的练习说明和逐行代码解释。
- `mlp_relu_demo.py`：主脚本，训练线性 baseline 和 MLP + ReLU。
- `outputs/training_log.txt`：保存两个模型的训练日志和样本预测。
- `outputs/linear_boundary.png`：线性模型学到的分类区域。
- `outputs/mlp_boundary.png`：MLP + ReLU 学到的分类区域。
- `notes.md`：解释线性边界、ReLU、MLP 和真实模型结构的关系。

## 先打开哪个文件

建议按这个顺序：

1. 先打开 `mlp_relu_demo.py`，看 `build_linear_model()` 和 `build_mlp_model()` 的区别。
2. 运行脚本，看两个模型的 accuracy 差异。
3. 打开 `outputs/linear_boundary.png` 和 `outputs/mlp_boundary.png`，直观看分类边界。
4. 最后打开 `outputs/training_log.txt`，对照 loss、accuracy 和样本预测。

```powershell
cd D:\1AIGC_daily_practice\day09_torch_mlp_relu
notepad mlp_relu_demo.py
```

## 安装依赖

今天需要 PyTorch 和 Pillow。如果报错缺少依赖：

```powershell
pip install torch pillow
```

## 运行命令

```powershell
cd D:\1AIGC_daily_practice\day09_torch_mlp_relu
python mlp_relu_demo.py
```

运行后打开输出：

```powershell
notepad outputs\training_log.txt
start outputs\linear_boundary.png
start outputs\mlp_boundary.png
```

## 应该观察什么输出

终端会打印：

```text
Linear baseline final accuracy：...
MLP + ReLU final accuracy：...
```

你要重点观察：

- 线性 baseline 在 XOR 数据上表现明显受限；
- MLP + ReLU 的 accuracy 应该接近 1.0；
- `linear_boundary.png` 的背景划分比较简单；
- `mlp_boundary.png` 能学出更适合 XOR 的非线性区域。

## 整体思路

这个练习做了一个对比实验：

```text
同一份 XOR 数据
-> 训练 Linear(2, 2)
-> 训练 Linear(2, 16) + ReLU + Linear(16, 2)
-> 比较 accuracy 和分类边界图
```

XOR 数据的规律是：

```text
左下 + 右上：same_sign
左上 + 右下：different_sign
```

这种分布不是一条直线能分开的，所以线性模型会吃力。MLP 多了隐藏层和 ReLU，能组合出更复杂的边界。

## 逐行解释代码

### 1. 导入库和类别名

```python
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image, ImageDraw

CLASS_NAMES = ["same_sign", "different_sign"]
```

逐行解释：

- `Path`：用于保存日志和图片。
- `torch`：生成数据、训练模型。
- `nn`：提供 `Linear`、`ReLU`、`Sequential`、`CrossEntropyLoss`。
- `DataLoader` / `TensorDataset`：按 mini-batch 读取数据。
- `Image` / `ImageDraw`：用 Pillow 画分类边界图。
- `CLASS_NAMES`：把类别编号映射成人能看懂的名字。

### 2. 构造 XOR 风格数据

```python
centers = torch.tensor([
    [-2.0, -2.0],
    [2.0, 2.0],
    [-2.0, 2.0],
    [2.0, -2.0],
])
labels_for_centers = torch.tensor([0, 0, 1, 1], dtype=torch.long)
```

逐行解释：

- 4 个中心点分别在左下、右上、左上、右下。
- 左下和右上标签是 0，也就是 `same_sign`。
- 左上和右下标签是 1，也就是 `different_sign`。
- 这种交叉分布就是 XOR 风格，线性模型很难用一条直线分开。

```python
points = center + torch.randn(samples_per_cluster, 2) * 0.45
class_labels = torch.full((samples_per_cluster,), label.item(), dtype=torch.long)
```

逐行解释：

- 在每个中心附近撒随机点。
- `0.45` 控制点云扩散程度。
- 每个中心生成一组对应标签。

### 3. DataLoader

```python
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
```

逐行解释：

- `TensorDataset(x, y)`：把坐标和标签绑定成样本。
- `batch_size=16`：每次取 16 个样本训练。
- `shuffle=True`：每轮打乱样本顺序。
- `generator`：固定打乱顺序，方便复现实验。

### 4. 线性 baseline

```python
def build_linear_model():
    torch.manual_seed(1)
    return nn.Linear(2, 2)
```

逐行解释：

- `nn.Linear(2, 2)`：输入二维点，输出 2 个类别 logits。
- 这个模型没有隐藏层，也没有 ReLU。
- 它只能学比较简单的线性分类边界。

### 5. MLP + ReLU

```python
def build_mlp_model(hidden_size=16):
    torch.manual_seed(1)
    return nn.Sequential(
        nn.Linear(2, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 2),
    )
```

逐行解释：

- `nn.Sequential(...)`：把多个层按顺序串起来。
- `nn.Linear(2, hidden_size)`：把二维输入映射到 16 个隐藏特征。
- `nn.ReLU()`：加入非线性。如果没有它，两层 Linear 叠起来仍然近似一个 Linear。
- `nn.Linear(hidden_size, 2)`：把隐藏特征映射成 2 个类别 logits。

### 6. 训练函数

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
```

逐行解释：

- `CrossEntropyLoss`：适合分类任务，直接接收 logits 和整数标签。
- `Adam`：比普通 SGD 更稳一些，适合这个小实验快速收敛。
- `model.parameters()`：把模型中所有可训练参数交给优化器。

```python
for batch_x, batch_y in dataloader:
    logits = model(batch_x)
    loss = loss_fn(logits, batch_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

逐行解释：

- `logits = model(batch_x)`：forward。
- `loss = loss_fn(...)`：计算分类损失。
- `zero_grad()`：清空旧梯度。
- `backward()`：计算梯度。
- `step()`：更新模型参数。

### 7. 画分类边界

```python
grid_tensor = torch.tensor(row_points, dtype=torch.float32)
predicted = model(grid_tensor).argmax(dim=1)
```

逐行解释：

- 把平面上密密麻麻的坐标点交给模型。
- 模型对每个点输出 logits。
- `argmax(dim=1)` 得到每个坐标点的预测类别。
- 背景颜色就是模型预测出来的区域。

圆点是真实训练样本，背景是模型预测区域。对比 `linear_boundary.png` 和 `mlp_boundary.png`，就能看到 ReLU 带来的非线性表达能力。

## 重点理解的 5 个代码点

### 1. `nn.Sequential`

它把多个层按顺序包装成一个模型。输入会依次经过每一层。

### 2. `nn.Linear`

线性层做的是：

```text
output = input @ weight.T + bias
```

### 3. `nn.ReLU`

ReLU 把负值截断为 0，引入非线性，让模型不再只是线性变换的叠加。

### 4. 隐藏层维度

`hidden_size=16` 表示中间有 16 个隐藏特征。它越大，模型表达能力通常越强，但也更容易过拟合。

### 5. 非线性边界

线性模型只能学线性边界；MLP + ReLU 可以组合出折线式、分段式的复杂边界。

## 常见错误

### 1. 忘记 ReLU

如果写成：

```python
nn.Linear(2, 16)
nn.Linear(16, 2)
```

中间没有激活函数，整体仍然接近线性模型，XOR 数据还是难分。

### 2. 输出维度和类别数不匹配

二分类要输出 2 个 logits：

```python
nn.Linear(hidden_size, 2)
```

### 3. label 类型错误

`CrossEntropyLoss` 需要 `torch.long` 类型的类别标签。

### 4. 学习率不合适

学习率太小会收敛慢，太大可能震荡。这个脚本里线性 baseline 用 Adam 和 `lr=0.003`，让它稳定停在约 50% 附近；MLP 用 Adam 和 `lr=0.03`，让你更快看到非线性模型收敛。

## 自查清单和参考答案

- 问：为什么线性 baseline 难以学好 XOR？
  答：因为 XOR 的两类点交叉分布，不能被一条线性边界完整分开。

- 问：ReLU 的作用是什么？
  答：引入非线性，让多层 Linear 不再等价于单个 Linear。

- 问：隐藏层 `16` 是类别数吗？
  答：不是。它是中间特征数量，类别数由最后一层输出维度决定。

- 问：为什么要看分类边界图？
  答：它能直观看出模型把平面划成了什么区域，比只看 accuracy 更容易理解模型能力。

## 可修改的小练习

### 练习 1：修改 hidden size

把：

```python
build_mlp_model(hidden_size=16)
```

改成：

```python
build_mlp_model(hidden_size=4)
```

参考结果：模型可能仍能学，但边界可能更粗糙，收敛也可能变慢。

### 练习 2：去掉 ReLU 对比

把 MLP 改成：

```python
nn.Sequential(
    nn.Linear(2, 16),
    nn.Linear(16, 2),
)
```

参考结果：它会更像线性模型，难以很好分开 XOR。

### 练习 3：把 optimizer 从 Adam 改成 SGD

把：

```python
torch.optim.Adam(model.parameters(), lr=lr)
```

改成：

```python
torch.optim.SGD(model.parameters(), lr=0.1)
```

参考结果：也可能学会，但收敛速度和稳定性会变化。

## 面试表达

可以这样总结今天的练习：

这个练习用 XOR 风格二维数据对比了线性分类器和 MLP。线性模型只能学习线性边界，难以分开交叉分布的数据；加入隐藏层和 ReLU 后，MLP 获得非线性表达能力，可以学出更复杂的分类区域。这个结构思想可以迁移到 CLIP projector、Transformer/DiT 的 FFN，以及 Stable Diffusion U-Net 中的 MLP/激活模块。
