# Day 09 Notes - MLP、ReLU 和非线性表达能力

今天的核心直觉是：线性模型只能学线性边界，而很多真实任务不是一条直线就能分开的。MLP 通过隐藏层和激活函数，把简单线性层组合成更强的非线性函数。

## 为什么线性模型不够

线性分类器做的事情可以理解成：

```text
用一条直线把二维平面分成两边
```

对于简单数据，这已经够用。但 XOR 风格数据很麻烦：

```text
左下 + 右上 是一类
左上 + 右下 是另一类
```

这两类交叉分布，没有一条直线能把它们完全分开。所以线性 baseline 的 accuracy 通常只能停在比较低的位置。

## ReLU 为什么能引入非线性

ReLU 的公式很简单：

```text
ReLU(x) = max(0, x)
```

它看起来只是把负数截断成 0，但它让模型不再只是多个线性层的叠加。

如果没有 ReLU：

```text
Linear -> Linear
```

整体仍然等价于一个 Linear。

有了 ReLU：

```text
Linear -> ReLU -> Linear
```

中间出现了“折一下”的非线性变换，模型就能组合出更复杂的分类边界。

## 隐藏层在做什么

今天的 MLP 是：

```python
nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
)
```

可以这样理解：

```text
输入 2 维坐标
-> 映射到 16 个隐藏特征
-> ReLU 保留有用激活、截断一部分负值
-> 再映射成 2 个类别 logits
```

隐藏层维度 `16` 表示模型有 16 个中间特征单元。它不是类别数，而是模型表达能力的一部分。

## MLP 在真实模型里对应哪里

在 CLIP 里，视觉特征和文本特征常会经过 projection 层映射到共同 embedding 空间。这个 projector 有时就是线性层或 MLP。

在 Transformer / DiT 里，每个 block 通常有 attention 和 FFN。FFN 本质上就是一种 MLP：

```text
Linear -> activation -> Linear
```

在 Stable Diffusion 的 U-Net 里，时间步 embedding、条件信息处理、某些 block 内部也会用到 MLP 或激活模块。

所以今天的 toy MLP 虽然小，但结构思想非常常见：

```text
线性映射 -> 非线性激活 -> 线性映射
```

## 模型能力由什么决定

模型不是只靠结构就能成功。至少要同时看：

- 结构复杂度：线性模型还是 MLP，隐藏层多大；
- 数据：数据是否有规律，标签是否可靠；
- loss：训练目标是否和任务匹配；
- optimizer：参数是否能被有效更新；
- 训练轮数和学习率：是否足够收敛。

今天的数据有明确 XOR 规律，MLP 有足够表达能力，CrossEntropyLoss 适合分类，所以 MLP 能明显超过线性 baseline。
