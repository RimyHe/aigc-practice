# Day 04 - NumPy 图像数组直觉

今天的目标是用最小代码理解一句非常重要的话：图像在模型里就是数字数组。你会看到一个 4x4 的 RGB 小图如何从 JSON 和 PNG 图片两种形式变成 NumPy 数组，如何查看 `shape` 和 `dtype`，如何把 0-255 的像素归一化到 0-1，以及为什么模型常需要把 HWC 转成 CHW。

## 今日目标

完成后你应该能说清楚：

1. 一张 RGB 图片为什么可以表示成三维数组；
2. `shape` 里的每个数字分别代表什么；
3. `dtype` 表示什么，为什么原始像素常是整数；
4. 为什么要做 `arr / 255.0` 归一化；
5. 如何用 Pillow 把真实图片文件读成像素矩阵；
6. 为什么 PyTorch 常用 CHW，而普通图像直觉常是 HWC。

## 文件结构

```text
day04_numpy_image_array/
  README.md
  data/
    tiny_rgb_array.json
    tiny_rgb_image.png
  outputs/
    array_summary.txt
  inspect_image_array.py
  notes.md
```

- `README.md`：今天的练习说明。
- `data/tiny_rgb_array.json`：一个 4x4 的 RGB 小图数组。
- `data/tiny_rgb_image.png`：由同一组 RGB 像素保存出来的小图片，用来演示图片文件如何转成像素矩阵。
- `outputs/array_summary.txt`：运行脚本后生成的数组统计结果。
- `inspect_image_array.py`：主脚本，读取 JSON 数组和 PNG 图片、打印 shape/dtype、归一化、计算通道均值、演示 HWC 到 CHW。
- `notes.md`：解释图像数组、HWC/CHW、归一化和 CLIP / Stable Diffusion / DiT 的联系。

## 先打开哪个文件

建议按这个顺序看：

1. 先打开 `data/tiny_rgb_array.json`，看 4x4 RGB 数组长什么样。
2. 再打开 `data/tiny_rgb_image.png`，直观看到这组数字对应的小图片。
3. 再打开 `inspect_image_array.py`，重点看 `Image.open`、`np.array`、`arr / 255.0`、`transpose(2, 0, 1)`。
4. 运行脚本，看终端输出。
5. 最后打开 `outputs/array_summary.txt`，确认结果已经保存。

可以用：

```powershell
cd D:\1AIGC_daily_practice\day04_numpy_image_array
notepad data\tiny_rgb_array.json
start data\tiny_rgb_image.png
notepad inspect_image_array.py
```

## 安装依赖

今天需要 NumPy 和 Pillow。如果运行时报 `No module named 'numpy'` 或 `No module named 'PIL'`，先安装：

```powershell
pip install numpy pillow
```

如果你的环境里已经有 NumPy 和 Pillow，就不需要重复安装。

## 运行命令

进入当天文件夹：

```powershell
cd D:\1AIGC_daily_practice\day04_numpy_image_array
```

运行脚本：

```powershell
python inspect_image_array.py
```

运行后打开输出文件：

```powershell
notepad outputs\array_summary.txt
```

## 应该观察什么输出

终端里应该看到类似：

```text
NumPy 图像数组检查结果
============================
原始数组 shape：(4, 4, 3)
原始数组 dtype：int64
原始数组最小值：0
原始数组最大值：255

示例像素位置：(y=0, x=0)
示例像素 RGB：R=255, G=0, B=0
解释：这个像素是纯红色，因为 R 很高，G 和 B 都是 0。

二、从 PNG 图片读取的像素矩阵
图片数组 shape：(4, 4, 3)
图片数组 dtype：uint8
图片数组最小值：0
图片数组最大值：255
图片同一位置 RGB：R=255, G=0, B=0
PNG 图片数组是否和 JSON 数组完全一致：True

归一化后最小值：0.000
归一化后最大值：1.000

每个通道的平均亮度，范围是 0 到 1：
- R 平均亮度：0.495
- G 平均亮度：0.517
- B 平均亮度：0.476

HWC shape：(4, 4, 3)
CHW shape：(3, 4, 4)
```

你要重点观察：

1. 原始 shape 是 `(4, 4, 3)`，说明它是 4 行、4 列、3 个颜色通道；
2. PNG 图片读进来后的 shape 也是 `(4, 4, 3)`，说明真实图片也会变成像素矩阵；
3. 归一化后最大值从 255 变成 1.0；
4. HWC 转 CHW 后，通道维从最后挪到了最前面。

## 整体思路

这个练习模拟的是图像进入深度学习模型前的最小预处理流程：

```text
JSON 里的 RGB 数字 -> NumPy 数组
PNG 图片文件 -> NumPy 数组
查看 shape/dtype -> 归一化 -> 调整通道顺序 -> 保存统计结果
```

具体执行流程：

1. 进入练习文件夹：

```powershell
cd D:\1AIGC_daily_practice\day04_numpy_image_array
```

2. 打开输入数据：

```powershell
notepad data\tiny_rgb_array.json
```

你会看到三层 list：

- 第一层：高度方向，有 4 行；
- 第二层：宽度方向，每行有 4 个像素；
- 第三层：每个像素的 RGB 三个值。

例如：

```json
[255, 0, 0]
```

表示红色像素，因为 R=255，G=0，B=0。

3. 打开真实图片：

```powershell
start data\tiny_rgb_image.png
```

这张图片很小，只有 4x4 像素。它和 JSON 里的数字是一一对应的：每一个小像素都能在 JSON 里找到对应的 RGB 三元组。

4. 运行脚本：

```powershell
python inspect_image_array.py
```

5. 脚本内部按这个顺序工作：

- 用 `Path(__file__).parent` 找到当前脚本所在目录；
- 拼出 `data/tiny_rgb_array.json`、`data/tiny_rgb_image.png` 和 `outputs/array_summary.txt`；
- 用 `json.load` 读取三层 list；
- 用 `np.array` 转成 NumPy 数组；
- 用 `Image.open(...).convert("RGB")` 打开 PNG 图片；
- 再用 `np.array(image)` 把图片转成像素矩阵；
- 对比 JSON 数组和 PNG 图片数组是否一致；
- 打印 `shape`、`dtype`、最小值、最大值；
- 取出 `(y=0, x=0)` 位置的像素，解释 RGB；
- 用 `arr / 255.0` 把像素值归一化到 0-1；
- 用 `mean(axis=(0, 1))` 计算每个通道的平均亮度；
- 用 `transpose(2, 0, 1)` 把 HWC 变成 CHW；
- 把统计结果保存到 `outputs/array_summary.txt`。

6. 打开输出文件：

```powershell
notepad outputs\array_summary.txt
```

真实工程里，检查 shape 和保存中间结果非常重要。很多模型报错不是算法本身错，而是输入形状、通道顺序或数值范围不对。

## 重点理解的 3 个代码点

### 1. `np.array`

代码：

```python
return np.array(raw_data)
```

`raw_data` 是从 JSON 里读出来的 Python list。list 可以存数据，但不适合高效做矩阵计算。`np.array` 会把它变成 NumPy 数组，让你可以方便地查看 shape、计算最小值最大值、做均值、做通道转换。

工程联系：真实 AIGC 项目里，图片、latent、embedding、attention map，本质上都会以数组或 tensor 的形式处理。NumPy 是你理解这些数据形状的第一步，PyTorch tensor 的很多直觉也和 NumPy 很像。

补充看图片读取：

```python
image = Image.open(image_path).convert("RGB")
image_arr = np.array(image)
```

这两行就是“真实图片文件 -> 像素矩阵”的关键。`Image.open` 打开图片，`convert("RGB")` 确保它有 R/G/B 三个通道，`np.array(image)` 把它变成可以计算的数组。

### 2. `arr / 255.0`

代码：

```python
arr_norm = arr / 255.0
```

原始 RGB 像素通常是 0 到 255 的整数。模型一般更喜欢范围稳定的浮点数，所以这里把它变成 0 到 1。

为什么除以 `255.0` 而不是 `255`？重点不是写 `255` 一定错，而是你要明确这是浮点归一化。结果应该是 0.0 到 1.0 之间的小数，而不是仍然停留在整数像素直觉里。

工程联系：CLIP、Stable Diffusion、DiT 的输入预处理都会涉及数值缩放或标准化。如果忘记归一化，模型看到的数值范围可能和训练时完全不同，输出质量会明显受影响。

### 3. `transpose(2, 0, 1)`

代码：

```python
arr_chw = arr_norm.transpose(2, 0, 1)
```

原始 `arr_norm` 的 shape 是：

```text
(4, 4, 3)
```

也就是 HWC：高度、宽度、通道。

`transpose(2, 0, 1)` 表示把原来的第 2 维通道放到最前面，变成：

```text
(3, 4, 4)
```

也就是 CHW：通道、高度、宽度。

工程联系：PyTorch 图像模型通常喜欢 CHW。如果你把 HWC 直接喂给模型，模型可能报 shape 错误，也可能把宽度当通道来理解，结果完全不对。

## 常见错误

### 1. 没安装 NumPy

报错可能是：

```text
ModuleNotFoundError: No module named 'numpy'
```

解决：

```powershell
pip install numpy
```

如果报的是：

```text
ModuleNotFoundError: No module named 'PIL'
```

解决：

```powershell
pip install pillow
```

### 2. shape 理解错

`(4, 4, 3)` 不是 3 张 4x4 图，而是一张 4x4 的 RGB 图。

参考答案：

```text
第一个 4：高度
第二个 4：宽度
3：RGB 三个通道
```

### 3. 整数除法和浮点归一化混淆

你要确认归一化后的值是小数，范围是 0 到 1。今天的代码：

```python
arr_norm = arr / 255.0
```

会得到浮点数组。后面做深度学习输入时，通常也需要浮点 tensor。

### 4. HWC / CHW 顺序混淆

HWC 是：

```text
高度、宽度、通道
```

CHW 是：

```text
通道、高度、宽度
```

今天的转换是：

```python
arr_norm.transpose(2, 0, 1)
```

不要把它误写成 `transpose(0, 2, 1)`，那会变成高度、通道、宽度，不是 PyTorch 常用格式。

## 自查清单和参考答案

- 问：`tiny_rgb_array.json` 的 shape 为什么是 `(4, 4, 3)`？
  答：因为它有 4 行，每行 4 个像素，每个像素有 R/G/B 三个通道。

- 问：`tiny_rgb_image.png` 读进来后为什么也是 `(4, 4, 3)`？
  答：因为这张 PNG 图片也是 4x4 的 RGB 图片。图片文件只是存储格式，读进程序后仍然会变成高度、宽度、通道组成的像素矩阵。

- 问：`[255, 0, 0]` 表示什么颜色？
  答：红色。R 通道最大，G 和 B 都是 0。

- 问：为什么要做 `arr / 255.0`？
  答：把 0-255 的整数像素缩放到 0-1 的浮点范围，让输入更接近深度学习模型常用的数据范围。

- 问：HWC 转成 CHW 后 shape 怎么变？
  答：从 `(4, 4, 3)` 变成 `(3, 4, 4)`，通道维从最后移动到最前面。

- 问：这个练习和 AIGC 实习有什么关系？
  答：AIGC 模型处理图像时，本质上处理的是数组或 tensor。CLIP 输入图像、Stable Diffusion 的 VAE latent、DiT 的 patch token，都离不开 shape、dtype、归一化和维度顺序。

## 可修改的小练习

### 练习 1：改一个像素颜色

目标：理解 RGB 数值如何对应颜色。

指引：打开 `data/tiny_rgb_array.json`，把第一个像素：

```json
[255, 0, 0]
```

改成：

```json
[0, 0, 255]
```

参考结果：再次运行后，示例像素会从红色变成蓝色。终端里仍会打印 `(y=0, x=0)` 的 RGB，但现在应该是 `R=0, G=0, B=255`。

### 练习 2：把 4x4 扩成 5x5

目标：理解 shape 随输入数据变化。

指引：在 JSON 里新增第 5 行，并确保每一行都有 5 个像素。原来的 4 行也要各补 1 个像素，否则形状会不规则。

参考新增思路：

```text
每行都从 4 个 RGB 像素变成 5 个 RGB 像素
总行数也从 4 行变成 5 行
```

参考结果：再次运行后，原始 shape 应该变成 `(5, 5, 3)`，CHW shape 应该变成 `(3, 5, 5)`。

### 练习 3：增加主色调判断

目标：练习用通道均值做简单判断。

指引：在 `build_summary()` 里 `channel_means = arr_norm.mean(axis=(0, 1))` 后面增加：

```python
dominant_index = int(np.argmax(channel_means))
dominant_channel = ["R", "G", "B"][dominant_index]
```

然后在 `lines` 里增加：

```python
f"平均亮度最高的通道：{dominant_channel}",
```

参考结果：默认数据里 G 通道均值最高，所以主色调判断会输出 `G`。

## 面试表达

可以这样总结今天的练习：

这个练习用一个 4x4 的 RGB 小图理解了图像数组的基本形式。我用 NumPy 查看 shape 和 dtype，把 0-255 像素归一化到 0-1，并演示了从 HWC 到 CHW 的通道顺序转换。这些概念是理解 PyTorch 图像输入、CLIP 预处理、Stable Diffusion 的 VAE latent，以及 DiT patch/token 表示的基础。
