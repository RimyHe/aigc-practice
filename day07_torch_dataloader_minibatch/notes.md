# Day 07 Notes - Dataset、DataLoader 和 mini-batch

今天的核心直觉是：真实训练通常不会一次把所有数据都塞进模型，而是把数据切成很多 mini-batch，一批一批送进模型。`Dataset` 负责“有哪些样本”，`DataLoader` 负责“每次取多少、是否打乱、如何迭代”。

## 为什么深度学习常用 mini-batch

如果一次只看一个样本，梯度噪声很大，训练可能抖动明显；如果一次看完整数据集，计算可能太慢、太占显存，也不利于频繁更新参数。

mini-batch 是折中方案：

```text
一个 batch 包含若干个样本
每次用一个 batch 做 forward / loss / backward / step
一个 epoch 遍历完整个数据集
```

今天有 100 个样本，`batch_size=10`，所以每个 epoch 有 10 个 batch。

## Dataset 和 DataLoader 分别做什么

`TensorDataset(x, y)` 把输入 `x` 和标签 `y` 绑定起来。第 0 个 `x` 对应第 0 个 `y`，第 1 个 `x` 对应第 1 个 `y`。

`DataLoader(dataset, batch_size=10, shuffle=True)` 负责按批次取数据：

- `batch_size=10`：每次取 10 条样本；
- `shuffle=True`：每个 epoch 打乱顺序；
- 迭代时返回 `(batch_x, batch_y)`。

## batch shape 怎么看

今天 `x` 的完整 shape 是：

```text
(100, 1)
```

表示 100 个样本，每个样本 1 个特征。

当 `batch_size=10` 时，一个 batch 的 shape 是：

```text
batch_x: (10, 1)
batch_y: (10, 1)
```

第一个维度 10 是 batch 维度，第二个维度 1 是特征或标签维度。

## DataLoader 如何对应真实项目

在线性回归里，样本是 `(x, y)`。

在图像分类里，样本可能是：

```text
(image_tensor, label)
```

在 CLIP 训练里，样本可能是：

```text
(image_tensor, text_tokens)
```

在 VAE 图像重建里，样本可能是：

```text
(image_tensor, image_tensor)
```

输入和目标都来自同一张图，模型学习重建。

在 Stable Diffusion 文生图训练里，样本可能包含：

```text
(latent, text_embedding, timestep, noise)
```

模型学习在给定文本条件和时间步时预测噪声。

在 DiT 训练里，样本可能是：

```text
(latent_patch_tokens, timestep, condition)
```

Transformer 处理的是 patch/token 序列，但训练循环仍然是一批一批读数据。

## 从 Day06 到 Day07 的变化

Day06 是全量训练：

```text
pred = model(x)
loss = loss_fn(pred, y)
```

每个 epoch 直接用全部 100 条样本。

Day07 是 mini-batch 训练：

```text
for batch_x, batch_y in dataloader:
    pred = model(batch_x)
    loss = loss_fn(pred, batch_y)
```

每个 epoch 内部还有一个 batch 循环。这就是今天最重要的结构变化：

```text
epoch 循环
  batch 循环
    forward
    loss
    zero_grad
    backward
    step
```
