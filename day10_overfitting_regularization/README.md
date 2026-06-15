# Day10：过拟合、weight decay 和 dropout

## 今日目标

今天你要跑通一个很小的 PyTorch 实验，亲眼看到：

```text
训练集 accuracy 很高，不代表模型真的好。
```

我们会构造一个二维 toy 二分类任务：

1. 训练集很小：只有 40 个点；
2. 测试集较大：400 个点；
3. 训练集加入 15% 标签噪声；
4. 模型故意做得偏大：两层 128 hidden size 的 MLP；
5. 对比三种训练方式：
   - baseline：没有正则；
   - weight decay：限制权重过大；
   - dropout：训练时随机关闭一部分神经元。

你今天最重要的观察不是“谁的训练集最高”，而是：

```text
train accuracy 和 test accuracy 差多少？
这个差距说明模型有没有过拟合？
```

## d2l PyTorch 对应参考

- 第 4.4 节：模型选择、欠拟合和过拟合  
  https://zh.d2l.ai/chapter_multilayer-perceptrons/underfit-overfit.html
- 第 4.5 节：权重衰减  
  https://zh.d2l.ai/chapter_multilayer-perceptrons/weight-decay.html
- 第 4.6 节：暂退法 Dropout  
  https://zh.d2l.ai/chapter_multilayer-perceptrons/dropout.html

## 文件结构

```text
day10_overfitting_regularization/
  README.md
  outputs/
    regularization_summary.txt
  regularization_demo.py
  notes.md
```

说明：

- `README.md`：今天先打开这个文件。
- `regularization_demo.py`：主脚本，训练三种模型并输出对比。
- `notes.md`：概念解释和 AIGC 工程迁移。
- `outputs/regularization_summary.txt`：运行脚本后生成的实验汇总。

## 先打开哪个文件

先打开：

```text
D:\1AIGC_daily_practice\day10_overfitting_regularization\README.md
```

然后打开：

```text
D:\1AIGC_daily_practice\day10_overfitting_regularization\regularization_demo.py
```

建议顺序：

1. 先读 README 的整体思路；
2. 跑脚本；
3. 看终端输出；
4. 打开 `outputs/regularization_summary.txt`；
5. 回来看逐行解释；
6. 最后做小练习。

## 运行命令

在 PowerShell 里运行：

```powershell
cd D:\1AIGC_daily_practice\day10_overfitting_regularization
python regularization_demo.py
```

如果提示没有安装 PyTorch，先安装：

```powershell
pip install torch
```

## 应该观察什么输出

你会先看到数据形状：

```text
训练集 x shape：(40, 2)
训练集 y shape：(40,)
测试集 x shape：(400, 2)
测试集 y shape：(400,)
```

含义：

- 训练集有 40 个二维点；
- 每个点有 2 个特征；
- 每个点有 1 个类别标签；
- 测试集更大，用来模拟“没见过的新数据”。

然后你会看到三组模型的训练日志：

```text
baseline
weight_decay
dropout
```

每行格式是：

```text
模型名 | epoch | train_loss | train_acc | test_acc | gap
```

其中：

- `train_loss`：训练集损失；
- `train_acc`：训练集准确率；
- `test_acc`：测试集准确率；
- `gap = train_acc - test_acc`：训练集和测试集差距。

你重点看：

```text
baseline 是否 train_acc 很高，但 test_acc 没那么高？
weight_decay / dropout 是否让 gap 变小，或者 test_acc 更稳？
```

## 整体思路

今天的实验可以按工程流程理解：

第一步：造数据。

```text
训练集：40 个点，有一点标签噪声。
测试集：400 个点，没有标签噪声。
```

为什么这样设计？

因为真实项目里，小数据经常有噪声。比如图文对不干净、caption 不准确、人工标签不一致。模型如果太大，就可能把这些噪声也记住。

第二步：定义一个偏大的模型。

```text
Linear(2, 128)
ReLU
Linear(128, 128)
ReLU
Linear(128, 2)
```

对于 40 个训练点来说，这个模型能力很强，足够记住训练集。

第三步：训练 baseline。

baseline 没有正则，模型可以尽量贴合训练集。它可能 train accuracy 很高，但 test accuracy 不一定好。

第四步：训练 weight decay 版本。

weight decay 会让优化器在更新参数时惩罚过大的权重。它的直觉是：

```text
不要为了记住训练集噪声，把参数调得太极端。
```

第五步：训练 dropout 版本。

dropout 在训练时随机关闭一部分神经元。它的直觉是：

```text
不要过度依赖某几个隐藏特征。
```

第六步：比较结果。

不要只看：

```text
train_acc 谁最高
```

要看：

```text
test_acc 谁更高
gap 谁更小
```

这就是从“会训练模型”走向“会评估模型”的关键一步。

## 重点理解的 5 个代码点

### 1. train/test split

代码中分别造了训练集和测试集：

```python
train_x, train_y = make_toy_data(num_samples=40, seed=0, label_noise=0.15)
test_x, test_y = make_toy_data(num_samples=400, seed=1, label_noise=0.0)
```

训练集用于更新参数，测试集只用于评估。

工程上必须这样做。否则你只知道模型有没有记住训练数据，不知道它能不能处理新数据。

### 2. overfitting gap

代码里计算：

```python
gap = train_acc - test_acc
```

如果：

```text
train_acc 很高
test_acc 明显低
gap 很大
```

这就是过拟合信号。

### 3. `weight_decay`

代码里：

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=weight_decay,
)
```

当 `weight_decay=1e-3` 时，优化器会限制权重过大。

工程直觉：

```text
参数不要太极端，模型不要太死记硬背。
```

### 4. `nn.Dropout`

代码里：

```python
if dropout_p > 0:
    layers.append(nn.Dropout(p=dropout_p))
```

当 `dropout_p=0.3` 时，训练阶段每次大约随机关闭 30% 的神经元激活。

这会迫使模型不要只靠某几个隐藏神经元完成任务。

### 5. `model.train()` / `model.eval()`

训练时：

```python
model.train()
```

评估时：

```python
model.eval()
```

这对 Dropout 非常重要。

- `model.train()`：Dropout 会随机丢弃神经元；
- `model.eval()`：Dropout 关闭随机丢弃，模型稳定输出。

如果测试时忘记 `model.eval()`，测试结果会带随机性，不可靠。

## 逐行解释代码

### 1. 导入依赖

```python
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
```

逐行解释：

- `Path`：用来稳定定位当前脚本和输出目录；
- `torch`：PyTorch 主库；
- `nn`：神经网络模块，比如 `Linear`、`ReLU`、`Dropout`；
- `DataLoader`：按 batch 读取训练数据；
- `TensorDataset`：把 `x` 和 `y` 包装成 PyTorch 数据集。

### 2. 构造 toy 数据

```python
def make_toy_data(num_samples, seed, label_noise=0.0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand((num_samples, 2), generator=generator) * 6.0 - 3.0
```

解释：

- `num_samples`：要生成多少个点；
- `seed`：固定随机种子，保证你每次运行结果可复现；
- `x` 的 shape 是 `(num_samples, 2)`；
- `torch.rand(...)` 原本生成 `[0, 1)` 的随机数；
- `* 6.0 - 3.0` 把范围变成 `[-3, 3)`。

```python
    score = x[:, 0] * x[:, 1] + 0.6 * torch.sin(1.5 * x[:, 0])
    y = (score > 0).long()
```

解释：

- `x[:, 0]` 是所有点的第 1 个坐标；
- `x[:, 1]` 是所有点的第 2 个坐标；
- `score` 是人为设计的非线性分类规则；
- `score > 0` 得到 True/False；
- `.long()` 把 True/False 转成类别标签 `0/1`。

```python
    if label_noise > 0:
        flip_mask = torch.rand(num_samples, generator=generator) < label_noise
        y[flip_mask] = 1 - y[flip_mask]
```

解释：

- `label_noise=0.15` 表示约 15% 标签会被翻转；
- 原本是 0 的变成 1；
- 原本是 1 的变成 0；
- 这是故意模拟真实数据里的错误标注。

### 3. 包装 DataLoader

```python
def make_dataloader(x, y, batch_size=16):
    dataset = TensorDataset(x, y)
```

解释：

- `TensorDataset(x, y)` 把输入和标签绑在一起；
- 第 `i` 个样本就是 `(x[i], y[i])`。

```python
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
```

解释：

- `batch_size=16`：每次取 16 个样本训练；
- `shuffle=True`：每轮打乱训练样本；
- `generator`：让打乱顺序也可复现。

### 4. 定义大 MLP

```python
def build_big_mlp(dropout_p=0.0):
    layers = [
        nn.Linear(2, 128),
        nn.ReLU(),
    ]
```

解释：

- 输入是二维点，所以第一层 `Linear(2, 128)`；
- 输出 128 个隐藏特征；
- `ReLU()` 加入非线性。

```python
    if dropout_p > 0:
        layers.append(nn.Dropout(p=dropout_p))
```

解释：

- baseline 和 weight decay 不加 dropout；
- dropout 版本会加入 `nn.Dropout(p=0.3)`。

```python
    layers.extend(
        [
            nn.Linear(128, 128),
            nn.ReLU(),
        ]
    )
```

解释：

- 再加一层隐藏层；
- 让模型容量更大；
- 小训练集 + 大模型更容易产生过拟合。

```python
    layers.append(nn.Linear(128, 2))
    return nn.Sequential(*layers)
```

解释：

- 最后一层输出 2 个 logits；
- 对应二分类的两个类别；
- `nn.Sequential(*layers)` 把列表里的层按顺序拼成模型。

### 5. 评估函数

```python
def evaluate(model, x, y, loss_fn):
    model.eval()
```

解释：

- 进入评估模式；
- Dropout 在这里会被关闭；
- 这一步非常重要。

```python
    with torch.no_grad():
        logits = model(x)
        loss = loss_fn(logits, y).item()
```

解释：

- 测试时不需要反向传播；
- `torch.no_grad()` 可以减少额外计算；
- `logits` 是模型输出的分类分数；
- `loss_fn(logits, y)` 计算交叉熵损失。

```python
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean().item()
```

解释：

- `argmax(dim=1)` 找每个样本分数最高的类别；
- `pred == y` 判断预测是否正确；
- `.mean()` 得到准确率。

### 6. 训练一个模型

```python
def train_one_model(..., weight_decay=0.0, dropout_p=0.0):
    torch.manual_seed(0)
    model = build_big_mlp(dropout_p=dropout_p)
```

解释：

- 每个模型都用同样随机种子初始化；
- 这样对比更公平；
- `dropout_p` 决定是否加入 Dropout。

```python
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        weight_decay=weight_decay,
    )
```

解释：

- `model.parameters()` 是要训练的权重和偏置；
- `lr=0.01` 是学习率；
- `weight_decay` 控制权重衰减强度；
- baseline 是 `0.0`；
- weight decay 版本是 `1e-3`。

```python
    for epoch in range(1, 101):
        model.train()
```

解释：

- 训练 100 个 epoch；
- 每轮开始前切换到训练模式；
- Dropout 只在训练模式生效。

```python
        for batch_x, batch_y in train_loader:
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
```

解释：

- 从 DataLoader 取一个 batch；
- 前向传播得到 logits；
- 用真实标签计算 loss。

```python
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

解释：

- `zero_grad()`：清空上一轮梯度；
- `backward()`：反向传播计算梯度；
- `step()`：根据梯度更新参数。

```python
        train_loss, train_acc = evaluate(model, train_x, train_y, loss_fn)
        test_loss, test_acc = evaluate(model, test_x, test_y, loss_fn)
        gap = train_acc - test_acc
```

解释：

- 每个 epoch 后同时评估训练集和测试集；
- `gap` 是过拟合的重要信号；
- gap 越大，越要警惕模型只是在记训练集。

### 7. 三种实验配置

```python
configs = [
    {"name": "baseline", "weight_decay": 0.0, "dropout_p": 0.0},
    {"name": "weight_decay", "weight_decay": 1e-3, "dropout_p": 0.0},
    {"name": "dropout", "weight_decay": 0.0, "dropout_p": 0.3},
]
```

解释：

- baseline：不做正则；
- weight decay：只加权重衰减；
- dropout：只加 Dropout；
- 三者模型主体相同，方便对比正则化影响。

## 常见错误

### 1. 只看训练集 accuracy

错误理解：

```text
train_acc = 1.0，所以模型很好。
```

正确理解：

```text
还要看 test_acc。如果 test_acc 低，说明泛化不好。
```

### 2. 不切换 `eval` 模式

如果测试时忘记：

```python
model.eval()
```

Dropout 还可能随机影响输出，评估结果就不稳定。

### 3. Dropout 放错位置

常见写法是放在隐藏层激活之后：

```python
Linear
ReLU
Dropout
```

不要放在最终 logits 后面，否则会直接随机破坏分类输出。

### 4. 测试时还反向传播

评估时应该使用：

```python
with torch.no_grad():
```

测试集不应该参与参数更新。

### 5. 学习率过大

学习率太大可能导致：

- loss 震荡；
- accuracy 不稳定；
- 正则化效果看不清。

## 自查清单和答案

### 1. 为什么要分训练集和测试集？

答案：训练集用来更新参数，测试集用来检查模型是否能处理没见过的数据。只看训练集会误判模型能力。

### 2. 什么现象说明过拟合？

答案：train accuracy 很高，但 test accuracy 明显低，train/test gap 很大。

### 3. weight decay 在限制什么？

答案：限制模型权重变得过大，降低模型为了贴合训练集噪声而出现极端参数的风险。

### 4. Dropout 训练和测试时一样吗？

答案：不一样。训练时随机丢弃一部分神经元；测试时关闭随机丢弃，所以必须用 `model.eval()`。

### 5. 为什么泛化能力比训练集 loss 更重要？

答案：真实项目面对的是新数据、新 prompt、新图片、新用户请求。训练集 loss 低只能说明模型会处理见过的数据，泛化能力才决定实际效果。

## 可修改的小练习

### 练习 1：改变训练集大小

把：

```python
train_x, train_y = make_toy_data(num_samples=40, seed=0, label_noise=0.15)
```

改成：

```python
train_x, train_y = make_toy_data(num_samples=100, seed=0, label_noise=0.15)
```

观察：

- baseline 的 test accuracy 是否变高；
- gap 是否变小。

参考理解：

训练数据更多时，模型不再只靠记住少数点，泛化通常会更稳。

### 练习 2：改变 dropout p

把：

```python
{"name": "dropout", "weight_decay": 0.0, "dropout_p": 0.3}
```

改成：

```python
{"name": "dropout", "weight_decay": 0.0, "dropout_p": 0.5}
```

观察：

- train accuracy 是否下降；
- test accuracy 是否更稳；
- gap 是否变小。

参考理解：

`p` 越大，随机关闭越多，正则越强。但太大也可能欠拟合。

### 练习 3：改变 weight_decay

把：

```python
{"name": "weight_decay", "weight_decay": 1e-3, "dropout_p": 0.0}
```

改成：

```python
{"name": "weight_decay", "weight_decay": 3e-3, "dropout_p": 0.0}
```

观察：

- train loss 是否变高；
- test loss 是否下降；
- train/test gap 是否变小。

参考理解：

更大的 weight decay 会更强地限制参数，但太大可能让模型学不动。

## 面试表达

你可以这样说：

```text
我做过一个 PyTorch 小实验，用小训练集和较大 MLP 制造过拟合现象。baseline 在训练集上准确率很高，但测试集 accuracy 明显低，说明模型记住了训练集噪声。然后我加入 weight decay 和 dropout，对比 train/test gap，理解了正则化不是为了让训练集分数最高，而是为了提升泛化稳定性。这和 CLIP 微调、Stable Diffusion LoRA、DiT 微调里的验证集评估逻辑是类似的。
```
