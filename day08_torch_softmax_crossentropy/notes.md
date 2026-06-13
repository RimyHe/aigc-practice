# Day 08 Notes - Logits、Softmax 和 Cross-Entropy

今天的核心直觉是：分类模型最后一层通常先输出 logits，而不是直接输出概率。logits 是未归一化分数；softmax 把这些分数变成概率分布；cross-entropy 用来衡量模型给正确类别分配的概率够不够高。

## logits 和概率的区别

logits 是模型直接输出的分数。它们可以是负数，也不要求加起来等于 1。

例如三分类模型可能输出：

```text
[2.1, -0.5, 0.3]
```

这只是 3 个类别的相对分数，还不是概率。

概率需要满足：

```text
每个值 >= 0
所有值加起来 = 1
```

softmax 会把 logits 转成这样的概率分布。

## softmax 为什么能变成概率

softmax 做两件事：

1. 对每个 logit 做指数运算，让结果变成正数；
2. 除以所有指数结果的总和，让它们加起来等于 1。

直觉上，logit 越大，对应类别的 softmax 概率就越大。softmax 不改变最大值所在的位置，所以预测类别常用：

```python
predicted = logits.argmax(dim=1)
```

也可以对概率做 argmax，结果一样。

## cross-entropy 为什么适合分类

分类任务关心的是：模型有没有把足够高的概率分给正确类别。

如果正确类别的概率很高，cross-entropy loss 就小；如果正确类别的概率很低，loss 就大。

这和分类目标很匹配：不仅要预测对，还希望模型对正确类别更有把握。

## 为什么 CrossEntropyLoss 直接吃 logits

PyTorch 的 `nn.CrossEntropyLoss()` 内部已经包含了 `log_softmax` 和负对数似然损失。

所以训练时应该写：

```python
loss = loss_fn(logits, labels)
```

不要写：

```python
loss = loss_fn(torch.softmax(logits, dim=1), labels)
```

后者会把概率再交给 CrossEntropyLoss，语义不对，也可能造成数值效果变差。

## 和 CLIP 的关系

CLIP 的图文对齐可以看成一个更大的分类问题。

假设一个 batch 里有 N 张图和 N 段文本。CLIP 会算出一个 N x N 的图文相似度矩阵：

```text
image_text_logits[i, j] = 第 i 张图和第 j 段文本的相似度
```

对第 i 张图来说，正确文本应该是第 i 段文本。因此每一行都可以看成一个 N 分类问题：

```text
在 N 段文本里，哪一段最匹配这张图？
```

训练时也会用 cross-entropy 让正确图文对的 logit 更高，让错误配对的 logit 更低。

## 和 text-to-image 检索的关系

如果你有一段文本和很多图片，可以先用模型算文本和每张图片的相似度 logits，再 softmax 或排序。

概率最高或 logit 最高的图片，就是模型认为最匹配文本的结果。

所以今天的 toy 三分类虽然很小，但它对应的核心思想可以迁移到：

- 图像分类；
- 文本分类；
- CLIP 图文匹配；
- text-to-image 检索；
- 多模态 reranking。

## 为什么画分类边界有帮助

二维 toy 数据最适合画图，因为每个样本就是平面上的一个点。

脚本里的 `classification_map.png` 做了两件事：

1. 把平面上很多密集坐标点都交给模型预测类别；
2. 用背景颜色表示每个位置被模型分到哪一类。

所以你看到的不是人工设定的中心点，而是训练后的线性分类器自己学到的划分区域。

圆点是真实训练样本，背景是模型预测区域。如果同色圆点大多落在同色背景里，说明模型已经学会了把这三类点分开。

这和真实图像分类的区别只是维度不同：二维点能画在平面上，图像 embedding 或 CLIP embedding 通常是高维向量，不能直接这么画，但分类逻辑仍然是“输出 logits，取最大分数对应的类别或匹配项”。
