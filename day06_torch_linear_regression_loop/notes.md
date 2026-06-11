# Day 06 Notes - 线性回归和完整训练循环

今天的核心直觉是：线性回归是深度学习训练循环的最小完整模型。它足够简单，只有一层线性关系；但它已经包含真实训练会反复出现的所有关键步骤：构造数据、模型预测、计算 loss、清空梯度、反向传播、参数更新。

## 线性回归为什么是最小模型

线性回归假设输入 `x` 和目标 `y` 之间近似满足：

```text
y = weight * x + bias
```

今天我们用的真实函数是：

```text
y = 3 * x + 2 + noise
```

其中 `noise` 是噪声。真实实验数据通常不会完美落在一条直线上，所以加噪声更贴近实际。

模型要学习的就是两个参数：

- `weight`：斜率，理想值接近 3；
- `bias`：偏置，理想值接近 2。

这比真实深度模型简单很多，但训练流程是同一个骨架。

## MSE loss 的含义

MSE 是 mean squared error，均方误差。它衡量预测值和真实值之间的平均平方差。

可以先这样理解：

```text
预测错得越多，loss 越大
预测越接近真实 y，loss 越小
```

平方有两个作用：

- 正误差和负误差都变成正数；
- 大错误会被放大，所以模型会努力减少明显偏离的预测。

在线性回归里，MSE 是最常用、最自然的 loss。

## optimizer 如何替代手动更新

Day05 里我们手动写过：

```python
with torch.no_grad():
    w -= lr * w.grad
```

Day06 用的是：

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
optimizer.step()
```

它们本质上做的是同一件事：根据梯度更新参数。区别是 optimizer 可以管理很多参数，不需要你手动一个个写 `weight -= lr * grad`。

真实模型里参数可能有几百万、几亿甚至更多，所以必须交给 optimizer 统一管理。

## 标准训练循环

今天脚本的核心顺序是：

```python
pred = model(x)
loss = loss_fn(pred, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

逐步理解：

- `pred = model(x)`：前向传播，用当前模型参数做预测。
- `loss = loss_fn(pred, y)`：计算预测和真实答案之间的差距。
- `optimizer.zero_grad()`：清空上一轮留下的梯度。
- `loss.backward()`：自动计算每个参数的梯度。
- `optimizer.step()`：根据梯度更新模型参数。

这个顺序以后会反复出现。无论是 CNN、CLIP、VAE、Stable Diffusion 还是 DiT，训练主干都离不开它。

## 映射到 CLIP、VAE、Stable Diffusion、DiT

在线性回归里：

```text
输入 x -> Linear 模型 -> pred -> MSE loss -> backward -> SGD 更新 weight/bias
```

在 CLIP 里：

```text
图像和文本 -> image/text encoder -> embedding -> 对比学习 loss -> backward -> optimizer 更新 encoder 参数
```

在 VAE 里：

```text
图像 -> encoder/decoder -> 重建图像 -> 重建 loss + KL loss -> backward -> optimizer 更新 VAE 参数
```

在 Stable Diffusion 里：

```text
latent + 文本条件 + 时间步 -> U-Net 预测噪声 -> 噪声预测 loss -> backward -> optimizer 更新 U-Net 或 LoRA 参数
```

在 DiT 里：

```text
patch/token 或 latent token -> Transformer -> 预测目标 -> loss -> backward -> optimizer 更新 Transformer 参数
```

所以今天的线性回归不是为了模型本身复杂，而是为了把训练循环真正跑通。你以后看大模型训练代码时，可以先问：输入在哪里，模型在哪里，loss 在哪里，zero_grad/backward/step 在哪里。
