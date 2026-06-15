# Day10 notes：过拟合、正则化和泛化能力

## 0. 阅读代码后的疑问

### 1. `eval` 模式是不是就是加入了 dropout 的版本？

不是。

`eval` 模式不是“加入 dropout”，而是“切换到评估状态”。

对 Dropout 来说：

```text
model.train()：Dropout 生效，训练时随机关闭一部分神经元。
model.eval()：Dropout 关闭，评估时使用完整网络稳定输出。
```

所以更准确的理解是：

```text
Dropout 是模型里的一个层；
train/eval 是模型当前处于训练还是评估状态；
Dropout 在 train 模式随机丢弃，在 eval 模式不随机丢弃。
```

如果模型里没有 `nn.Dropout`，`model.train()` 和 `model.eval()` 仍然存在，只是它们对 Dropout 没有影响。它们也会影响 BatchNorm 这类层。

### 2. `layers = [nn.Linear(2, 128), nn.ReLU()]` 代表只有两层吗？

这段代码代表当前列表里先放了两个模块：

```python
layers = [
    nn.Linear(2, 128),
    nn.ReLU(),
]
```

它不是整个模型的全部结构，因为后面还会继续 `append` 和 `extend`：

```python
layers.append(nn.Dropout(p=dropout_p))
layers.extend([nn.Linear(128, 128), nn.ReLU()])
layers.append(nn.Linear(128, 2))
```

最终 baseline 的结构是：

```text
Linear(2, 128)
ReLU
Linear(128, 128)
ReLU
Linear(128, 2)
```

如果是 dropout 版本，中间还会多两个 Dropout：

```text
Linear(2, 128)
ReLU
Dropout
Linear(128, 128)
ReLU
Dropout
Linear(128, 2)
```

按“模块数”数，baseline 有 5 个模块；按“有参数的层”数，baseline 有 3 个 Linear 层；按“隐藏层”数，它有 2 个隐藏层。

### 3. 什么叫“打乱顺序也可以复现”？

`shuffle=True` 表示每个 epoch 训练前会打乱样本顺序。

例如原始顺序可能是：

```text
0, 1, 2, 3, 4, 5
```

打乱后可能变成：

```text
3, 0, 5, 1, 4, 2
```

“可以复现”的意思是：虽然顺序被打乱了，但只要随机种子一样，每次运行时打乱出来的顺序仍然一样。

代码里：

```python
generator = torch.Generator().manual_seed(0)
```

就是给 DataLoader 的打乱过程指定一个固定随机源。这样你今天运行、明天运行、换一次参数后再运行，数据顺序不会莫名其妙变化，实验对比更公平。

### 4. 为什么 seed 固定就可以让实验复现？

计算机里的很多“随机”其实是伪随机。它不是完全不可控的随机，而是从一个初始数字开始，按固定算法生成一串看起来随机的数。

这个初始数字就是 seed。

如果 seed 一样，随机序列就一样：

```text
seed = 0 -> 第 1 个随机数、第 2 个随机数、第 3 个随机数都固定
```

在这个脚本里，seed 会影响：

- 训练点和测试点怎么生成；
- 哪些训练标签被翻转；
- 模型初始权重；
- DataLoader 每轮怎么打乱样本。

固定 seed 的工程意义是：

```text
我改了 weight_decay 之后，结果变化主要来自 weight_decay，而不是来自随机初始化或随机数据顺序。
```

注意：在 GPU、大规模并行训练里，完全复现还会涉及 CUDA、cuDNN、分布式训练等设置。这里是 CPU 小实验，固定 seed 已经足够帮助你理解可复现。

### 5. weight decay 是不是就是正则化？

weight decay 是正则化的一种，但不是正则化的全部。

更准确地说：

```text
正则化：一类防止模型过拟合的方法。
weight decay：一种常见正则化方法，限制权重过大。
dropout：另一种常见正则化方法，训练时随机关闭一部分神经元。
```

所以关系是：

```text
正则化
├── weight decay
├── dropout
├── 数据增强
├── early stopping
└── 其他限制模型过拟合的方法
```

在 PyTorch 里：

```python
torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
```

这就是通过优化器参数加入 weight decay 正则。

### 6. 为什么 Dropout 要出现两次？这两次隐藏的特征一样吗？

因为这个模型有两个隐藏层，每个隐藏层后面都可以做一次 Dropout。

Day10 的 dropout 版本结构是：

```text
Linear(2, 128)
ReLU
Dropout
Linear(128, 128)
ReLU
Dropout
Linear(128, 2)
```

第一次 Dropout 作用在第一层隐藏特征上：

```text
输入二维点 -> Linear(2, 128) -> ReLU -> 128 个第一层隐藏特征
```

这 128 个特征更接近“从原始二维坐标里提取出的基础判断”。例如某些神经元可能学到点在左边、右边、某条斜线的一侧等。

第二次 Dropout 作用在第二层隐藏特征上：

```text
第一层隐藏特征 -> Linear(128, 128) -> ReLU -> 128 个第二层隐藏特征
```

这 128 个特征已经不是原始坐标的直接判断，而是基于第一层特征重新组合出来的更高层表示。

所以两次 Dropout 隐藏的不是同一批特征。

更具体地说：

```text
第一次 Dropout：随机屏蔽第一层 ReLU 后的 128 个激活值中的一部分。
第二次 Dropout：随机屏蔽第二层 ReLU 后的 128 个激活值中的一部分。
```

即使两个隐藏层维度都是 128，它们的含义也不同。第一层的第 10 个神经元和第二层的第 10 个神经元不是同一个东西，它们属于不同层，有不同的权重、不同的输入来源，也学习不同的特征。

工程直觉：

```text
如果只在第一层 Dropout，模型的后半部分仍可能过度依赖第二层里的某些组合特征。
如果两层都 Dropout，就同时约束基础特征和组合特征，正则更强。
```

但 Dropout 也不是越多越好。如果每一层都加很大的 Dropout，模型可能学不动，出现欠拟合。真实项目里通常需要调 `p`，比如 `0.1`、`0.3`、`0.5` 对比验证集效果。

### 7. 如果理论上神经网络由输入层、隐藏层和输出层组成，本项目如何体现？

Day10 的模型可以对应到经典结构：

```text
输入层 -> 隐藏层 -> 输出层
```

不过在 PyTorch 代码里，输入层通常不是一个单独的 `nn.Linear`，而是由输入数据的 shape 体现。

本项目的输入数据是：

```python
x = torch.rand((num_samples, 2), generator=generator) * 6.0 - 3.0
```

所以每个样本有 2 个特征：

```text
输入层：2 维输入，也就是一个二维点 [x1, x2]
```

模型结构是：

```text
Linear(2, 128)
ReLU
Dropout
Linear(128, 128)
ReLU
Dropout
Linear(128, 2)
```

对应关系：

```text
输入层：x 的 shape 是 (batch_size, 2)
隐藏层 1：Linear(2, 128) 产生 128 维隐藏表示
隐藏层 2：Linear(128, 128) 产生新的 128 维隐藏表示
输出层：Linear(128, 2) 产生 2 个 logits
```

`ReLU` 和 `Dropout` 通常不单独算新的“层级角色”，它们是隐藏层里的处理模块：

```text
ReLU：给隐藏表示加入非线性
Dropout：训练时随机屏蔽一部分隐藏激活
```

所以这个项目可以说：

```text
输入层有 2 个特征；
有 2 个隐藏层，每个隐藏层 128 维；
输出层有 2 个 logits；
代码里一共有 3 个 Linear 模块、2 个 ReLU、2 个 Dropout。
```

### 8. 为什么把训练集从 40 改成 100 后，正则化效果反而看起来变差？

因为实验条件变了。原始设置是：

```text
训练集：40 个点
测试集：400 个点
训练集有 15% 标签噪声
```

这个设置故意制造：

```text
小数据 + 大模型 + 标签噪声
```

所以 baseline 很容易记住训练集，出现：

```text
train_acc 很高
test_acc 明显低
gap 很大
```

当训练集改成 100 个点后，训练数据覆盖更充分，baseline 也更容易学到真实规律，而不是只记住少数点。此时过拟合条件被削弱，正则化的优势自然不明显。

如果训练集仍然有 `label_noise=0.15`，还可能出现：

```text
train_acc 低于 test_acc
```

这不一定是错误。因为训练集里有一部分标签被故意翻转，是更“脏”的数据；测试集没有标签噪声，反而更干净。模型学到真实规律后，可能在干净测试集上表现更好。

工程理解：

```text
正则化不是保证每次 test accuracy 都提升；
它是在过拟合风险较强时，帮助模型减少死记硬背。
```

当 baseline 已经泛化很好时，再加 weight decay 或 dropout 可能会稍微限制模型能力，结果反而不如 baseline。

### 9. 过拟合和参数数量有关吗？

有关。参数数量越多，模型越“能记”，所以在数据少、噪声多、训练太久、正则弱的时候，更容易过拟合。

但不能简单说：

```text
参数多 = 一定过拟合
```

更准确是：

```text
参数多 + 数据少 + 噪声多 + 训练久 + 正则弱 -> 更容易过拟合
```

Day10 里输入只有 2 维，训练集只有 40 个点，但模型用了：

```text
Linear(2, 128)
Linear(128, 128)
Linear(128, 2)
```

对于这么小的数据来说，模型容量偏大，所以 baseline 很容易把训练集记住。

工程判断不要只看参数数量，而要看：

```text
train_acc 很高
test_acc 明显低
gap 很大
```

如果 train/test gap 不大，即使参数多，也不一定出现明显过拟合。

### 10. 参数数值大小为什么和过拟合有关？weight decay 为什么能缓解？

这里的“参数大小”指的是权重数值本身，比如某些 weight 变得很大。

线性层可以写成：

```text
output = input @ weight.T + bias
```

如果权重很大，模型输出会对输入的小变化非常敏感。

一维例子：

```text
y = w * x + b
```

如果 `w = 1`，输入从 `1.0` 变到 `1.1`，输出只变 `0.1`。

如果 `w = 100`，输入从 `1.0` 变到 `1.1`，输出会变 `10`。

这说明：

```text
参数数值越大，模型越容易变得陡峭、敏感、极端。
```

在分类任务里，如果训练集有噪声，模型可能为了把几个错误标签也分对，把权重调得很大，让分类边界弯得很厉害。这会让训练集表现很好，但测试集表现变差。

weight decay 的作用是给训练目标加一个偏好：

```text
不只要分类 loss 小，也希望参数不要太大。
```

可以粗略理解为：

```text
总损失 = 原本的分类 loss + lambda * 参数平方和
```

如果参数变得很大，平方项会快速变大，模型就会受到惩罚。

所以 weight decay 会让模型倾向于：

```text
使用较小的权重；
学习更平滑的边界；
不要为了少数噪声点过度扭曲。
```

这就是为什么 Day10 里可能出现：

```text
baseline：train_acc 更高，但 test_acc 较低
weight_decay：train_acc 略低，但 test_acc 更稳
```

注意：weight decay 不是魔法。太小约束不够，太大可能让模型学不动，变成欠拟合。

## 1. 欠拟合是什么

欠拟合是模型能力不够，连训练集都学不好。

工程上常见表现：

- train loss 高；
- train accuracy 低；
- test accuracy 也低。

这通常说明模型太简单、训练不够、特征不够，或者学习率等训练设置不合适。

例子：用一条直线去拟合很复杂的非线性边界，就可能欠拟合。

## 2. 过拟合是什么

过拟合是模型把训练集里的细节、噪声、偶然样本都记住了，但换到新数据上表现不好。

工程上常见表现：

- train accuracy 很高；
- test accuracy 明显低；
- train/test gap 很大。

本练习里训练集只有 40 个点，而且故意加入了 15% 标签噪声。一个 2 层、128 hidden size 的大 MLP 有足够能力把这些小数据记住，所以 baseline 很容易出现过拟合。

## 3. 泛化能力是什么

泛化能力指模型在没见过的数据上的表现。

真实项目里，你真正关心的通常不是：

```text
模型在训练集上多漂亮
```

而是：

```text
模型在验证集、测试集、线上真实请求里是否稳定
```

AIGC 场景尤其明显。一个 LoRA 在训练图上还原得很好，不代表它在新 prompt、新姿态、新风格上就好。

## 4. weight decay 是什么

weight decay 通常也叫 L2 正则。它会惩罚过大的权重，让模型不要为了记住训练集噪声而把参数调得很极端。

在 PyTorch 里常写成：

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=1e-3,
)
```

工程直觉：

```text
没有 weight decay：模型可以非常用力地贴合训练集。
有 weight decay：模型被提醒“别把参数搞得太夸张”。
```

它不保证每次都让 test accuracy 最高，但通常能降低过拟合风险，让模型更稳。

## 5. dropout 是什么

Dropout 会在训练时随机关闭一部分神经元，例如：

```python
nn.Dropout(p=0.3)
```

意思是训练时每次大约有 30% 的激活被置为 0。

它的工程直觉是：

```text
不要让模型过度依赖某几个隐藏特征；
迫使模型学到更分散、更鲁棒的表达。
```

注意：Dropout 只应该在训练时随机丢弃；测试时要关闭随机丢弃，所以必须理解：

```python
model.train()
model.eval()
```

## 6. 为什么大模型更容易记住小数据

参数越多，模型能表达的函数越复杂。小训练集只有几十个样本时，大模型可能不只是学到规律，还能把每个样本的偶然性都记下来。

这在工程里很常见：

- 小数据微调大模型；
- prompt/图片样本太少；
- 数据标签不干净；
- 训练轮数太多；
- 学习率和正则设置不合适。

## 7. 和 CLIP 微调的关系

CLIP 微调时，如果图文对数据很少，模型可能记住训练集里的特定图片和 caption，而不是学到通用图文对齐。

这时要关注：

- train retrieval accuracy；
- validation retrieval accuracy；
- 是否只在训练 caption 上表现好；
- 新图、新文本上的检索是否稳定。

## 8. 和 VAE / Stable Diffusion LoRA 的关系

LoRA 微调很容易过拟合小数据。

常见现象：

- 训练图还原很好；
- 新 prompt 泛化差；
- 人脸、服装、风格被过度绑定；
- 换姿态、换光照后崩坏。

缓解方式包括：

- 减少训练步数；
- 使用 weight decay；
- 调低学习率；
- 做数据增强；
- 增加正则图像或更丰富 prompt；
- 用验证 prompt 做人工评估。

## 9. 和 DiT 微调的关系

DiT / Transformer 类模型参数量大，表达能力强。小数据微调时，如果不控制正则和验证集，很容易训练集 loss 很漂亮，但生成质量没有真正提升。

你以后看实验日志时要养成习惯：

```text
不要只看 train loss；
要看 validation loss、生成样例、人工评价、CLIP score、FID 或业务指标。
```

## 10. 和 AIGC 产品侧评估的关系

产品侧更关心“真实用户输入下是否稳定”。

一个模型在训练集或 demo prompt 上表现好，不代表上线稳定。你需要准备：

- 训练集；
- 验证集；
- 测试 prompt 集；
- 边界 case；
- 人工评价表；
- 不同版本模型的对比记录。

今天这个小实验就是把这件事缩小到最小可运行版本：

```text
训练集很小
模型很大
baseline 记住训练集
正则化尝试改善泛化
最终看 train/test gap
```
