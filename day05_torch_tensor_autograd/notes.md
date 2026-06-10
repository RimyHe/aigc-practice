# Day 05 Notes - Tensor、梯度和最小训练循环

今天的核心直觉是：NumPy array 主要负责数值计算，而 PyTorch Tensor 不只是数组，它还可以记录计算过程，并自动计算梯度。这就是深度学习训练能自动更新参数的基础。

## Tensor 和 NumPy array 的区别

NumPy array 可以表示矩阵、图片、向量，也能做加减乘除和统计计算。Day04 里我们已经用它查看图像的 shape、dtype、归一化和通道顺序。

PyTorch Tensor 很像 NumPy array，但多了几个面向深度学习的能力：

- 可以在 CPU 或 GPU 上计算；
- 可以和神经网络层一起使用；
- 可以设置 `requires_grad=True`；
- 可以通过 autograd 自动求梯度。

可以先这样理解：

```text
NumPy array：适合通用数值处理
PyTorch Tensor：适合模型训练和深度学习计算
```

## 为什么训练模型需要梯度

训练模型的目标是让 loss 变小。loss 可以理解成“模型预测和目标答案之间差得有多远”。

但模型参数很多，程序需要知道：

```text
哪个参数应该变大？
哪个参数应该变小？
每次应该改多少？
```

梯度就是回答这个问题的信号。比如今天的例子：

```python
loss = (w * x - y) ** 2
```

`w.grad` 表示 loss 对参数 `w` 的导数。它告诉我们：如果改变 `w`，loss 会怎样变化。训练时常用的更新方向是：

```text
w = w - learning_rate * gradient
```

这就是最小版的梯度下降。

## autograd 如何对应真实 PyTorch 训练

今天脚本里的训练循环是：

```text
forward：pred = w * x
loss：loss = (pred - y) ** 2
backward：loss.backward()
update：w -= lr * w.grad
zero grad：w.grad.zero_()
```

真实 PyTorch 训练也是这个骨架，只是规模更大：

```text
输入 batch -> 模型 forward -> 计算 loss -> backward -> optimizer.step() -> optimizer.zero_grad()
```

今天我们没有用 `torch.optim`，而是手动写：

```python
with torch.no_grad():
    w -= lr * w.grad
```

这样可以更清楚地看到参数到底是怎么更新的。

## 为什么要清空梯度

PyTorch 默认会累积梯度。也就是说，如果你连续多次调用 `loss.backward()`，新的梯度会加到旧梯度上。

训练循环里通常每一步只希望用当前 batch 的梯度更新一次，所以更新参数后要清空梯度：

```python
w.grad.zero_()
```

真实训练里常见写法是：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## 和 Stable Diffusion / CLIP / DiT 的关系

今天的例子只有一个参数 `w`，但真实模型有大量参数。

在 CLIP 训练里，模型会让图像 embedding 和文本 embedding 更接近。loss 衡量图文是否匹配，backward 会计算 image encoder 和 text encoder 里大量参数的梯度。

在 Stable Diffusion 训练或微调里，模型常学习预测噪声。loss 衡量预测噪声和真实噪声之间的差距，backward 会把梯度传回 U-Net、文本条件模块或 LoRA 参数。

在 DiT 里，Transformer 处理图像 patch 或 latent token。训练时同样是 forward 得到预测，算 loss，然后 backward 得到梯度并更新参数。

所以今天这个小式子：

```python
loss = (w * x - y) ** 2
```

不是为了模拟完整模型，而是让你抓住训练的最小骨架：

```text
参数 -> 预测 -> loss -> 梯度 -> 更新参数 -> loss 下降
```
