# Day 08 - PyTorch Softmax 和 Cross-Entropy

今天的目标是理解分类任务里的三个关键词：`logits`、`softmax`、`cross-entropy`。你会训练一个最小三分类模型：输入二维点，输出 3 个 logits，然后用 `CrossEntropyLoss` 学会把不同区域的点分到不同类别。

## 今日目标

完成后你应该能说清楚：

1. 模型输出的 logits 为什么不是概率；
2. softmax 如何把 logits 转成概率分布；
3. 为什么 `CrossEntropyLoss` 训练时要直接传 logits；
4. `argmax` 如何从 logits 或概率里得到预测类别；
5. accuracy 如何衡量分类模型是否预测正确。

## d2l PyTorch 对应参考

本练习对齐《动手学深度学习》PyTorch 版本：

- [3.4 softmax 回归](https://zh.d2l.ai/chapter_linear-networks/softmax-regression.html)：理解 logits、softmax、交叉熵、accuracy。
- [3.6 softmax 回归的从零开始实现](https://zh.d2l.ai/chapter_linear-networks/softmax-regression-scratch.html)：理解手写 softmax、交叉熵和训练流程。
- [3.7 softmax 回归的简洁实现](https://zh.d2l.ai/chapter_linear-networks/softmax-regression-concise.html)：对应本练习里的 `nn.Linear`、`CrossEntropyLoss` 和 PyTorch 简洁训练写法。

注意：本练习只使用 PyTorch 写法，不使用 MXNet、TensorFlow 或 Paddle 写法。

## 文件结构

```text
day08_torch_softmax_crossentropy/
  README.md
  outputs/
    training_log.txt
    classification_map.png
  softmax_crossentropy_demo.py
  notes.md
```

- `README.md`：今天的练习说明和逐行代码解释。
- `softmax_crossentropy_demo.py`：主脚本，生成三分类数据并训练最小分类模型。
- `outputs/training_log.txt`：运行后保存训练前后预测和每轮训练日志。
- `outputs/classification_map.png`：运行后生成的二维分类可视化图，展示模型如何划分坐标平面。
- `notes.md`：解释 logits、softmax、cross-entropy 以及它们和 CLIP 图文对齐的关系。

## 先打开哪个文件

建议按这个顺序：

1. 先打开 `softmax_crossentropy_demo.py`，看 `make_dataset()`、`describe_samples()` 和 `train_model()`。
2. 再运行脚本，看训练前后的 logits / probabilities / prediction。
3. 最后打开 `outputs/training_log.txt`，对照 loss 和 accuracy 的变化。
4. 再打开 `outputs/classification_map.png`，直观看模型如何把坐标平面划成 3 类。

```powershell
cd D:\1AIGC_daily_practice\day08_torch_softmax_crossentropy
notepad softmax_crossentropy_demo.py
```

## 安装依赖

今天需要 PyTorch。如果运行时报 `No module named 'torch'`，先安装：

```powershell
pip install torch
```

脚本还会用 Pillow 保存分类图。如果运行时报 `No module named 'PIL'`，安装：

```powershell
pip install pillow
```

## 运行命令

```powershell
cd D:\1AIGC_daily_practice\day08_torch_softmax_crossentropy
python softmax_crossentropy_demo.py
```

运行后打开日志：

```powershell
notepad outputs\training_log.txt
```

打开分类图：

```powershell
start outputs\classification_map.png
```

## 应该观察什么输出

脚本会先打印数据形状：

```text
样本总数：60
x shape：(60, 2)
y shape：(60,)
```

含义是：60 个样本，每个样本 2 个特征；标签是一维整数类别。

然后会打印训练前样本预测。重点看：

```text
logits=[...]
probs=[...]
pred=...
```

训练后你应该看到 accuracy 接近 1.0，并且样本的 softmax 概率明显更偏向正确类别。

还会生成一张分类图：

```text
outputs/classification_map.png
```

图里有两类信息：

- 背景颜色：模型认为这个坐标区域属于哪一类；
- 圆点：真实训练样本，颜色表示真实标签。

如果模型学得好，同色圆点应该基本落在同色背景区域里。不同背景颜色交界的地方，就是模型学到的分类边界。

## 整体思路

这个练习模拟最小多分类训练：

```text
生成三类二维点 -> DataLoader 按 batch 读取 -> Linear(2, 3) 输出 logits -> CrossEntropyLoss -> SGD 更新 -> 观察概率、accuracy 和分类边界
```

核心训练循环：

```python
logits = model(batch_x)
loss = loss_fn(logits, batch_y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

注意：`CrossEntropyLoss` 直接接收 logits，不要先手动 softmax。

## 逐行解释代码

下面像老师带读一样，把 `softmax_crossentropy_demo.py` 拆开。

### 1. 导入库和类别名

```python
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CLASS_NAMES = ["red", "green", "blue"]
```

逐行解释：

- `Path`：用来保存 `outputs/training_log.txt`。
- `torch`：生成数据、计算 softmax、训练模型。
- `nn`：提供 `Linear` 和 `CrossEntropyLoss`。
- `DataLoader` / `TensorDataset`：按 mini-batch 组织训练数据。
- `CLASS_NAMES`：把类别编号 0、1、2 映射成人能读懂的名字。

### 2. 构造 toy 三分类数据

```python
centers = torch.tensor([
    [-2.0, -2.0],
    [2.0, 0.0],
    [0.0, 2.0],
])
```

逐行解释：

- 这里定义 3 个类别中心，每个中心是二维坐标。
- 第一类在左下，第二类在右侧，第三类在上方。
- 模型要学的是：二维点落在哪个区域，就属于哪个类别。

```python
points = center + torch.randn(samples_per_class, 2) * 0.45
class_labels = torch.full((samples_per_class,), class_id, dtype=torch.long)
```

逐行解释：

- `torch.randn(..., 2)`：围绕类别中心生成二维随机点。
- `* 0.45`：控制噪声大小。
- `torch.full(...)`：给这一类所有点创建同一个类别标签。
- `dtype=torch.long`：分类标签必须是整数 long 类型，`CrossEntropyLoss` 需要这种标签。

### 3. DataLoader

```python
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, generator=generator)
```

逐行解释：

- `TensorDataset(x, y)`：把特征和标签绑成样本。
- `batch_size=16`：每次取 16 个样本。
- `shuffle=True`：训练时打乱样本顺序。
- `generator`：固定打乱顺序，方便复现实验。

### 4. accuracy

```python
predicted = logits.argmax(dim=1)
correct = (predicted == labels).sum().item()
return correct / labels.numel()
```

逐行解释：

- `logits.argmax(dim=1)`：对每一行 logits 找最大值所在类别。
- `predicted == labels`：判断预测是否等于真实标签。
- `.sum().item()`：统计预测正确的数量。
- `labels.numel()`：标签总数。
- 正确数除以总数，就是 accuracy。

### 5. 训练前后查看 logits 和概率

```python
logits = model(sample_x)
probs = torch.softmax(logits, dim=1)
predicted = probs.argmax(dim=1)
```

逐行解释：

- `model(sample_x)`：得到每个样本对 3 个类别的 logits。
- `torch.softmax(logits, dim=1)`：沿类别维度把 logits 转成概率。
- `argmax(dim=1)`：选出概率最大的类别。

这里的 softmax 是为了观察概率，不是为了传给 loss。

### 6. 模型、loss 和 optimizer

```python
model = nn.Linear(2, 3)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
```

逐行解释：

- `nn.Linear(2, 3)`：输入是二维点，输出是 3 个类别的 logits。
- `CrossEntropyLoss()`：内部已经包含 log-softmax 和负对数似然。
- `SGD`：根据梯度更新模型参数。

### 7. 训练循环

```python
for batch_x, batch_y in dataloader:
    logits = model(batch_x)
    loss = loss_fn(logits, batch_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

逐行解释：

- `batch_x` shape 是 `(batch_size, 2)`。
- `logits` shape 是 `(batch_size, 3)`。
- `batch_y` shape 是 `(batch_size,)`，每个值是类别编号。
- `loss_fn(logits, batch_y)`：直接传 logits 和整数标签。
- 后三行是标准训练步骤：清梯度、反向传播、更新参数。

### 8. 画出分类区域

```python
for py in range(height):
    coord_y = y_max - (py / (height - 1)) * (y_max - y_min)
    row_points = []
    for px in range(width):
        coord_x = x_min + (px / (width - 1)) * (x_max - x_min)
        row_points.append([coord_x, coord_y])

    grid_tensor = torch.tensor(row_points, dtype=torch.float32)
    predicted = model(grid_tensor).argmax(dim=1)
```

逐行解释：

- `height` 和 `width` 是输出图片尺寸。
- `px`、`py` 是图片里的像素坐标。
- `coord_x`、`coord_y` 是把图片像素位置转换回真实二维坐标。
- `row_points.append([coord_x, coord_y])`：把一整行坐标点收集起来。
- `model(grid_tensor)`：让模型判断这些坐标点属于哪一类。
- `argmax(dim=1)`：选择 logit 最大的类别，作为该坐标位置的预测类别。

这一步的作用是：不仅看训练样本怎么分类，还看整个平面每个位置会被模型分到哪一类。

```python
pixels[px, py] = tuple(int(0.82 * 255 + 0.18 * c) for c in base_color)
```

逐行解释：

- `base_color` 是类别颜色，比如 red / green / blue。
- 这里把类别颜色调淡，作为背景色。
- 背景色表示模型学到的分类区域。

```python
for point, label in zip(x, y):
    px, py = to_pixel(point.tolist())
    color = CLASS_COLORS[label.item()]
    draw.ellipse(...)
```

逐行解释：

- 遍历真实训练样本；
- 把真实坐标转换成图片像素坐标；
- 根据真实标签选择颜色；
- 用圆点画出来。

所以最终图像可以这样读：

```text
背景 = 模型预测的分类区域
圆点 = 真实训练样本
圆点颜色和背景颜色一致 = 模型分对了
```

## 重点理解的 5 个代码点

### 1. logits

logits 是模型原始输出分数，不要求非负，也不要求总和为 1。

### 2. softmax

```python
probs = torch.softmax(logits, dim=1)
```

softmax 把每一行 logits 转成概率分布。

### 3. CrossEntropyLoss

```python
loss = loss_fn(logits, batch_y)
```

PyTorch 的 `CrossEntropyLoss` 直接吃 logits，不要先手动 softmax。

### 4. argmax

```python
predicted = logits.argmax(dim=1)
```

选出分数最高的类别作为预测。

### 5. accuracy

accuracy 是预测正确数量除以总样本数。它比 loss 更直观，但 loss 才是训练优化的目标。

### 6. 分类边界

分类图里不同背景颜色交界的地方，就是模型学到的分类边界。因为模型是 `Linear(2, 3)`，所以边界大致是直线切分出来的区域。

## 常见错误

### 1. 把 softmax 后的概率再传入 CrossEntropyLoss

错误写法：

```python
loss = loss_fn(torch.softmax(logits, dim=1), labels)
```

正确写法：

```python
loss = loss_fn(logits, labels)
```

### 2. 标签不是 long 类型

`CrossEntropyLoss` 需要整数类别标签，通常是 `torch.long`。

### 3. 输出维度和类别数不匹配

三分类任务必须输出 3 个 logits：

```python
nn.Linear(2, 3)
```

### 4. batch 维度理解错

`logits` shape 是 `(batch_size, num_classes)`，不是 `(num_classes, batch_size)`。

### 5. 没有生成图片

如果没有看到 `classification_map.png`，先确认脚本是否完整运行结束，并检查：

```powershell
Get-ChildItem outputs
```

如果报 `No module named 'PIL'`，需要安装 Pillow：

```powershell
pip install pillow
```

## 自查清单和参考答案

- 问：为什么模型输出 3 个数？
  答：因为这是三分类任务，每个类别对应一个 logit。

- 问：softmax 后的概率加起来是多少？
  答：每个样本对应的一行概率加起来是 1。

- 问：为什么训练时不把 softmax 结果传给 CrossEntropyLoss？
  答：因为 PyTorch 的 CrossEntropyLoss 内部已经包含 log-softmax。

- 问：accuracy 和 loss 的区别是什么？
  答：accuracy 是预测对的比例，loss 是优化目标，能反映模型对正确类别的置信程度。

- 问：分类图里的背景颜色表示什么？
  答：表示模型认为该坐标区域属于哪一类。

- 问：圆点颜色表示什么？
  答：表示训练样本的真实类别标签。

## 可修改的小练习

### 练习 1：改学习率

把：

```python
lr=0.2
```

改成：

```python
lr=0.05
```

参考结果：accuracy 仍会上升，但收敛可能更慢。

### 练习 2：查看训练前后概率变化

指引：重点对比日志里的“训练前样本预测”和“训练后样本预测”。

参考结果：训练后正确类别的概率通常会明显变大。

### 练习 3：改类别数

增加第 4 个中心点，同时把：

```python
CLASS_NAMES = ["red", "green", "blue"]
model = nn.Linear(2, 3)
```

改成 4 个类别和 `nn.Linear(2, 4)`。

参考结果：每个样本会输出 4 个 logits，softmax 后得到 4 类概率。

## 面试表达

可以这样总结今天的练习：

这个练习用一个 toy 三分类任务理解了 logits、softmax 和 cross-entropy。模型最后一层输出每个类别的 logits，softmax 只用于把它解释成概率，而训练时 PyTorch 的 `CrossEntropyLoss` 直接接收 logits 和整数标签。这个机制可以迁移到 CLIP 的图文相似度矩阵：每一行相似度都可以看成一个分类问题，用 cross-entropy 让正确图文对的分数更高。
