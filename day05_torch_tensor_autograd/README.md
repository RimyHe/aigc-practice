# Day 05 - PyTorch Tensor 和 Autograd

今天的目标是从 NumPy 数组过渡到 PyTorch Tensor，理解一句非常关键的话：Tensor 不只是数组，它还能记录计算图并自动求梯度。你会用一个只有一个参数 `w` 的极小问题，看清楚训练循环最核心的 4 步：forward、loss、backward、update。

## 今日目标

完成后你应该能说清楚：

1. `torch.tensor` 和 NumPy array 的直觉关系；
2. `requires_grad=True` 为什么表示“这个量需要被训练”；
3. `loss.backward()` 为什么能自动算出 `w.grad`；
4. 为什么更新参数时要写 `with torch.no_grad()`；
5. 为什么每一步更新后要清空梯度。

## 文件结构

```text
day05_torch_tensor_autograd/
  README.md
  outputs/
    training_log.txt
  tensor_autograd_demo.py
  notes.md
```

- `README.md`：今天的练习说明。
- `tensor_autograd_demo.py`：主脚本，演示 Tensor、autograd 和 10 步最小优化。
- `outputs/training_log.txt`：运行脚本后生成的训练日志。
- `notes.md`：解释 Tensor、梯度、autograd 和真实模型训练之间的关系。

## 先打开哪个文件

建议按这个顺序看：

1. 先打开 `tensor_autograd_demo.py`，重点看 `run_one_step_demo()` 和 `run_training_loop()`。
2. 再运行脚本，看终端里的 loss、grad、w 如何变化。
3. 最后打开 `outputs/training_log.txt`，把每一步训练记录对照代码看。

可以用：

```powershell
cd D:\1AIGC_daily_practice\day05_torch_tensor_autograd
notepad tensor_autograd_demo.py
```

## 安装依赖

今天需要 PyTorch。如果运行时报 `No module named 'torch'`，说明当前环境没有安装 PyTorch。

普通安装方式通常是：

```powershell
pip install torch
```

如果安装很慢或失败，先不要卡在这里。你可以把报错贴出来，我们再根据你的 Python 版本、系统和网络环境处理。

## 运行命令

进入当天文件夹：

```powershell
cd D:\1AIGC_daily_practice\day05_torch_tensor_autograd
```

运行脚本：

```powershell
python tensor_autograd_demo.py
```

运行后打开输出文件：

```powershell
notepad outputs\training_log.txt
```

## 应该观察什么输出

终端里会先看到普通 Tensor 的信息：

```text
普通 tensor：
tensor([[1., 2.],
        [3., 4.]])
shape：(2, 2)
dtype：torch.float32
```

然后看到单步 autograd：

```text
初始参数 w：1.000000
x：2.000000
目标 y：10.000000
预测 pred = w * x：2.000000
loss = (pred - y) ** 2：64.000000
w.grad：-32.000000
手动更新后 w：4.200000
```

最后看到 10 步优化过程。你要重点观察：

1. loss 会从很大逐步变小；
2. `w` 会从 1.0 逐步接近 5.0；
3. 因为 `x=2.0, y=10.0`，所以理想关系是 `w * 2 = 10`，也就是 `w = 5`。

## 整体思路

这个练习模拟的是训练循环的最小核心：

```text
参数 w -> forward 得到预测 -> 计算 loss -> backward 得到梯度 -> 手动更新 w -> 清空梯度
```

具体执行流程：

1. 进入练习文件夹：

```powershell
cd D:\1AIGC_daily_practice\day05_torch_tensor_autograd
```

2. 打开脚本：

```powershell
notepad tensor_autograd_demo.py
```

3. 先看普通 Tensor：

```python
normal_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
```

这一步对应你从 NumPy 数组过渡到 Tensor。它也是一个数字矩阵，有 shape 和 dtype。

4. 再看可训练参数：

```python
w = torch.tensor(1.0, requires_grad=True)
```

这里的 `w` 不是普通数字，而是一个需要梯度的 Tensor。它像模型里的一个参数。

5. 看 forward 和 loss：

```python
pred = w * x
loss = (pred - y) ** 2
```

这里的问题是：希望 `w * 2.0` 接近 `10.0`。如果初始 `w=1.0`，预测就是 2.0，离 10.0 很远，所以 loss 很大。

6. 看 backward：

```python
loss.backward()
```

PyTorch 会沿着前面的计算过程自动求导，把 loss 对 `w` 的导数放到：

```python
w.grad
```

7. 看参数更新：

```python
with torch.no_grad():
    w -= lr * w.grad
```

这一步就是手动梯度下降。`lr` 是学习率，`w.grad` 是更新方向。

8. 看清空梯度：

```python
w.grad.zero_()
```

训练循环里每一步都要清空旧梯度，否则梯度会累积，导致更新不符合你预期。

## 逐行解释代码

这一节我们像老师带读一样，把 `tensor_autograd_demo.py` 拆开看。你先不用背 API，重点是抓住每一行在训练流程里的位置。

### 1. 导入库

```python
from pathlib import Path

import torch
```

逐行解释：

- `from pathlib import Path`：导入路径处理工具。脚本后面要把训练日志保存到 `outputs/training_log.txt`，用 `Path` 拼路径比手写字符串更稳。
- `import torch`：导入 PyTorch。今天所有 Tensor、自动求梯度、参数更新都依赖它。

工程联系：真实训练脚本里也会先导入路径工具、PyTorch、数据处理库、模型定义等。导入部分相当于告诉你“这个脚本会用哪些工具”。

### 2. 格式化 Tensor 数值

```python
def format_tensor_value(value):
    return f"{value.item():.6f}"
```

逐行解释：

- `def format_tensor_value(value):`：定义一个小工具函数，用来把 Tensor 里的单个数值格式化成固定小数位。
- `value.item()`：从只包含一个数的 Tensor 中取出 Python 数字。比如 `tensor(1.0)` 取出来就是 `1.0`。
- `:.6f`：表示保留 6 位小数，方便日志对齐观察。

注意：`.item()` 适合打印和记录，不适合在训练计算中乱用。训练计算要尽量保留 Tensor 形式，这样 PyTorch 才能追踪计算图。

### 3. 单步演示函数的开头

```python
def run_one_step_demo(lines):
    lines.append("一、单步 autograd 演示")
    lines.append("-" * 32)
```

逐行解释：

- `def run_one_step_demo(lines):`：定义“单步演示”函数。它不直接写文件，而是把要输出的文字追加到 `lines` 这个列表里。
- `lines.append(...)`：往日志列表里加一行文本。最后这些文本会同时打印到终端和保存到文件。
- `"-" * 32`：生成 32 个横线，用来让输出更清楚。

工程联系：训练脚本里经常会记录日志。今天用 `lines` 列表做最小版日志系统，真实项目可能用 logging、TensorBoard、WandB 或 CSV。

### 4. 创建普通 Tensor

```python
normal_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
lines.append(f"普通 tensor：\n{normal_tensor}")
lines.append(f"shape：{tuple(normal_tensor.shape)}")
lines.append(f"dtype：{normal_tensor.dtype}")
lines.append("")
```

逐行解释：

- `torch.tensor([[1.0, 2.0], [3.0, 4.0]])`：创建一个 2 行 2 列的 Tensor。它像 NumPy array 一样，是一个数字矩阵。
- `normal_tensor.shape`：查看形状。这里是 `(2, 2)`，表示 2 行 2 列。
- `normal_tensor.dtype`：查看数据类型。这里通常是 `torch.float32`，表示 32 位浮点数。
- `lines.append("")`：加一个空行，让输出更容易读。

工程联系：Day04 里图像数组有 shape 和 dtype；到了 PyTorch，图片、embedding、latent、模型参数也都有 shape 和 dtype。读深度学习代码时，先看 shape 是基本功。

### 5. 创建最小训练问题

```python
x = torch.tensor(2.0)
y = torch.tensor(10.0)
w = torch.tensor(1.0, requires_grad=True)
lr = 0.1
```

逐行解释：

- `x = torch.tensor(2.0)`：输入是 2.0。你可以把它理解成一条极小训练样本的输入。
- `y = torch.tensor(10.0)`：目标答案是 10.0。训练目标是让模型输出接近它。
- `w = torch.tensor(1.0, requires_grad=True)`：创建参数 `w`，初始值是 1.0。`requires_grad=True` 表示 PyTorch 要记录围绕 `w` 的计算，并能对它求梯度。
- `lr = 0.1`：学习率，控制每次沿梯度方向更新多少。

这 4 行对应真实训练里的四类东西：

- `x`：输入数据，比如图片、文本 token、latent；
- `y`：目标，比如类别标签、目标 embedding、真实噪声；
- `w`：模型参数；
- `lr`：优化器超参数。

### 6. Forward、loss、backward

```python
pred = w * x
loss = (pred - y) ** 2
loss.backward()
```

逐行解释：

- `pred = w * x`：forward。用当前参数 `w` 对输入 `x` 做预测。初始时 `w=1.0`，`x=2.0`，所以 `pred=2.0`。
- `loss = (pred - y) ** 2`：计算损失。预测是 2.0，目标是 10.0，差距是 -8，平方后 loss 是 64。
- `loss.backward()`：反向传播。PyTorch 会沿着 `loss -> pred -> w` 这条计算链，自动计算 loss 对 `w` 的导数，并存到 `w.grad`。

核心直觉：forward 是“用参数算结果”，backward 是“根据结果差距反推参数该怎么改”。

### 7. 记录单步结果

```python
lines.append(f"初始参数 w：{format_tensor_value(w)}")
lines.append(f"x：{format_tensor_value(x)}")
lines.append(f"目标 y：{format_tensor_value(y)}")
lines.append(f"预测 pred = w * x：{format_tensor_value(pred)}")
lines.append(f"loss = (pred - y) ** 2：{format_tensor_value(loss)}")
lines.append(f"w.grad：{format_tensor_value(w.grad)}")
lines.append("解释：w.grad 表示 loss 对 w 的导数，也就是 w 往哪个方向改会影响 loss。")
```

逐行解释：

- 这些 `lines.append(...)` 都是在写日志，不改变训练本身。
- `format_tensor_value(w)`：把 Tensor 里的单个数取出来，格式化成 `1.000000` 这样的形式。
- `w.grad`：这是最重要的观察对象。它不是参数本身，而是 loss 对参数 `w` 的梯度。

为什么 `w.grad` 是负数？因为当前 `w` 太小，预测 `w*x` 只有 2，目标是 10。要让 loss 变小，`w` 应该变大。梯度下降公式是 `w = w - lr * grad`，当 grad 是负数时，减去负数就会让 `w` 增大。

### 8. 手动更新参数

```python
with torch.no_grad():
    w -= lr * w.grad
```

逐行解释：

- `with torch.no_grad():`：告诉 PyTorch，这一小段不要记录进计算图。
- `w -= lr * w.grad`：手动梯度下降。学习率 `lr=0.1`，梯度 `w.grad=-32`，所以更新量是 `0.1 * -32 = -3.2`。执行 `w -= -3.2` 后，`w` 从 1.0 变成 4.2。

为什么更新参数要放在 `no_grad()` 里？因为参数更新是训练控制逻辑，不是 forward 计算本身。如果 PyTorch 把“更新参数”也记录进计算图，后续求梯度会变复杂甚至报错。

### 9. 清空梯度

```python
lines.append(f"手动更新后 w：{format_tensor_value(w)}")
w.grad.zero_()
lines.append(f"清空梯度后 w.grad：{format_tensor_value(w.grad)}")
lines.append("")
```

逐行解释：

- 第一行记录更新后的 `w`。
- `w.grad.zero_()`：把梯度清零。末尾带 `_` 的 PyTorch 方法通常表示“原地修改”，这里就是直接把 `w.grad` 变成 0。
- 后两行继续写日志。

为什么要清空？PyTorch 默认会累积梯度。如果不清空，下一次 `loss.backward()` 得到的新梯度会叠加到旧梯度上，训练行为就不再是“每一步用当前 loss 更新一次”。

### 10. 10 步训练循环的开头

```python
def run_training_loop(lines):
    lines.append("二、10 步最小优化过程")
    lines.append("-" * 32)

    x = torch.tensor(2.0)
    y = torch.tensor(10.0)
    w = torch.tensor(1.0, requires_grad=True)
    lr = 0.1
```

逐行解释：

- `def run_training_loop(lines):`：定义 10 步训练函数。
- 前两行 `lines.append`：写日志标题。
- `x`、`y`、`w`、`lr`：重新创建一套输入、目标、参数和学习率。这样 10 步循环从干净的 `w=1.0` 开始，不受前面单步演示影响。

工程联系：真实训练里也会先初始化模型参数、准备数据、设置学习率，然后进入训练循环。

### 11. 循环里的 forward、loss、backward

```python
for step in range(1, 11):
    pred = w * x
    loss = (pred - y) ** 2
    loss.backward()
```

逐行解释：

- `for step in range(1, 11):`：循环 10 次，`step` 从 1 到 10。
- `pred = w * x`：每一步都用当前最新的 `w` 重新计算预测。
- `loss = (pred - y) ** 2`：计算当前预测和目标之间的差距。
- `loss.backward()`：根据当前 loss 计算当前梯度。

注意：每一步的 `w` 都会变化，所以每一步的 `pred`、`loss`、`grad` 也会变化。这就是训练过程。

### 12. 取出日志数值

```python
w_before = w.item()
grad = w.grad.item()
loss_value = loss.item()
```

逐行解释：

- `w_before = w.item()`：记录更新前的参数值。
- `grad = w.grad.item()`：记录当前梯度。
- `loss_value = loss.item()`：记录当前 loss。

这些值用于写日志。它们不会参与后续训练计算，所以可以用 `.item()` 转成普通 Python 数字。

### 13. 更新参数并记录一行训练日志

```python
with torch.no_grad():
    w -= lr * w.grad

w_after = w.item()
lines.append(
    f"{step:>4} | {w_before:>8.4f} | {loss_value:>8.4f} | "
    f"{grad:>8.4f} | {w_after:>8.4f}"
)

w.grad.zero_()
```

逐行解释：

- `with torch.no_grad():`：参数更新不进入计算图。
- `w -= lr * w.grad`：按梯度下降更新参数。
- `w_after = w.item()`：记录更新后的参数值。
- `lines.append(...)`：把当前 step 的 `w_before`、`loss`、`grad`、`w_after` 写成一行表格。
- `w.grad.zero_()`：清空梯度，准备下一步。

你可以把这一段看成真实训练循环里的最小手写版：

```text
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

今天不用 optimizer，是为了让你亲眼看见 `w -= lr * grad` 这件事。

### 14. 保存日志

```python
def save_log(log_text, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(log_text)
```

逐行解释：

- `def save_log(log_text, output_path):`：定义保存日志的函数。
- `output_path.parent.mkdir(parents=True, exist_ok=True)`：确保 `outputs/` 文件夹存在。如果没有，就自动创建。
- `with open(..., "w", encoding="utf-8") as f:`：用 UTF-8 打开文件，准备写入中文日志。
- `f.write(log_text)`：把完整训练日志写入 `training_log.txt`。

工程联系：训练模型时，日志非常重要。你需要知道 loss 有没有下降、参数有没有异常、输出保存在哪里。没有日志，就很难复盘实验。

### 15. 主入口 `main()`

```python
def main():
    project_dir = Path(__file__).parent
    output_path = project_dir / "outputs" / "training_log.txt"

    lines = [
        "PyTorch Tensor 与 Autograd 最小演示",
        "=" * 40,
        "",
    ]

    run_one_step_demo(lines)
    run_training_loop(lines)

    log_text = "\n".join(lines)
    save_log(log_text, output_path)

    print(log_text)
    print()
    print(f"训练日志已保存到：{output_path}")
```

逐行解释：

- `def main():`：主入口。读脚本时先看它，就能知道程序整体做什么。
- `project_dir = Path(__file__).parent`：找到当前脚本所在目录。
- `output_path = project_dir / "outputs" / "training_log.txt"`：拼出输出日志路径。
- `lines = [...]`：创建日志列表，先放标题。
- `run_one_step_demo(lines)`：执行单步 autograd 演示。
- `run_training_loop(lines)`：执行 10 步训练循环。
- `"\n".join(lines)`：把很多行字符串合成一个大字符串，中间用换行连接。
- `save_log(log_text, output_path)`：保存到文件。
- `print(log_text)`：同时打印到终端。
- 最后一个 `print`：告诉你日志保存到了哪里。

整体来看，`main()` 串起了完整流程：

```text
准备输出路径 -> 创建日志 -> 单步演示 -> 多步训练 -> 保存日志 -> 打印结果
```

### 16. 入口保护

```python
if __name__ == "__main__":
    main()
```

逐行解释：

- `if __name__ == "__main__":`：判断这个文件是不是被直接运行。
- `main()`：如果是直接运行，就执行主流程。

为什么要这样写？因为一个 Python 文件既可以直接运行，也可以被别的文件导入。加上这两行后，别人以后如果只想导入 `run_training_loop()` 或 `save_log()`，不会自动跑完整训练流程。

## 重点理解的 4 个代码点

### 1. `torch.tensor`

代码：

```python
normal_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
```

它和 NumPy array 一样能表示数字矩阵，也有 `shape` 和 `dtype`。区别是 PyTorch Tensor 可以进入神经网络训练流程，支持 GPU，也支持自动求梯度。

工程联系：图像、文本 embedding、latent、模型参数，在 PyTorch 里基本都是 Tensor。

### 2. `requires_grad=True`

代码：

```python
w = torch.tensor(1.0, requires_grad=True)
```

这表示 `w` 需要参与梯度计算。你可以把它理解成“这个值是要被训练更新的参数”。

工程联系：真实模型里的权重参数，比如线性层、卷积层、attention 层参数，本质上都是需要梯度的 Tensor。

### 3. `loss.backward()`

代码：

```python
loss.backward()
```

这一步会从 loss 出发，沿着计算图反向传播，计算每个可训练参数对 loss 的影响。今天只有一个参数 `w`，所以结果在 `w.grad` 里。

工程联系：CLIP、Stable Diffusion、DiT 训练时，backward 会自动计算大量参数的梯度。你不用手写每一层的导数。

### 4. `with torch.no_grad()`

代码：

```python
with torch.no_grad():
    w -= lr * w.grad
```

更新参数这一步不应该被 PyTorch 继续记录进计算图，所以要放在 `no_grad()` 里。

工程联系：真实训练里通常用 `optimizer.step()` 更新参数，它内部会处理这类细节。今天手动写，是为了让你看清楚更新到底发生了什么。

## 常见错误

### 1. 没安装 PyTorch

报错可能是：

```text
ModuleNotFoundError: No module named 'torch'
```

可以先试：

```powershell
pip install torch
```

### 2. 忘记清空梯度

如果不写：

```python
w.grad.zero_()
```

梯度会累积。你会发现每一步的更新不再只是当前 loss 的梯度，而是叠加了之前的梯度。

### 3. 在 `no_grad` 外原地更新参数

不要直接在普通计算图环境里写：

```python
w -= lr * w.grad
```

这可能让 PyTorch 把参数更新也记录进计算图，导致报错或后续梯度行为变复杂。手动更新参数时，放进：

```python
with torch.no_grad():
```

### 4. 把 Tensor 和 Python float 混淆

`w` 是 Tensor，不是普通 Python 数字。打印单个数值时可以用：

```python
w.item()
```

但训练计算时应该尽量保留 Tensor 形式，才能让 PyTorch 追踪计算图。

## 自查清单和参考答案

- 问：为什么理想的 `w` 接近 5？
  答：因为目标是让 `w * x` 接近 `y`。这里 `x=2.0`，`y=10.0`，所以 `w * 2 = 10`，得到 `w = 5`。

- 问：`w.grad` 表示什么？
  答：它表示 loss 对 `w` 的导数，也就是改变 `w` 会怎样影响 loss。

- 问：为什么 loss 会下降？
  答：因为每一步都按 `w = w - lr * grad` 更新，方向是让 loss 变小的方向。

- 问：为什么要清空梯度？
  答：PyTorch 默认累积梯度。清空后，下一步的梯度才只来自当前这一次 forward 和 loss。

- 问：这个例子和真实训练有什么关系？
  答：真实训练也是 forward、loss、backward、update，只是参数从一个 `w` 变成大量模型权重，输入从一个数变成 batch 数据。

## 可修改的小练习

### 练习 1：修改 `x` 和 `y`

目标：观察最优 `w` 如何变化。

指引：把训练循环里的：

```python
x = torch.tensor(2.0)
y = torch.tensor(10.0)
```

改成：

```python
x = torch.tensor(3.0)
y = torch.tensor(12.0)
```

参考结果：理想 `w` 会从 5.0 变成 4.0，因为 `w * 3 = 12`。

### 练习 2：修改学习率

目标：观察学习率对收敛速度的影响。

指引：把：

```python
lr = 0.1
```

改成：

```python
lr = 0.01
```

参考结果：loss 仍然会下降，但速度会慢很多，10 步后 `w` 可能还没接近 5.0。

### 练习 3：把循环步数从 10 改到 30

目标：观察更多训练步后是否更接近目标。

指引：把：

```python
for step in range(1, 11):
```

改成：

```python
for step in range(1, 31):
```

参考结果：日志会变成 30 步，`w` 会更接近 5.0，`w * x` 会更接近 10.0，loss 会更接近 0。

## 面试表达

可以这样总结今天的练习：

这个练习用一个只有一个参数的最小例子理解了 PyTorch 训练循环。我创建了带 `requires_grad=True` 的 Tensor，通过 forward 计算预测和 loss，用 `loss.backward()` 自动得到梯度，再用梯度下降手动更新参数。它对应真实模型训练里的 forward、loss、backward、optimizer update 这条主线。
