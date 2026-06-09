# Day 04 Notes - 图像为什么是数组

今天的核心直觉是：模型看到的图像不是“图片”这个概念，而是一堆数字。我们平时看到的 RGB 图片，可以理解成一个三维数组。

## 从图片文件到像素矩阵

图片文件，比如 PNG 或 JPG，本质上是把像素信息按某种格式保存到了磁盘上。人双击看到的是图片，但程序读取后，会把它还原成像素矩阵。

今天脚本里用的是：

```python
image = Image.open(image_path).convert("RGB")
image_arr = np.array(image)
```

可以这样理解：

```text
tiny_rgb_image.png -> Pillow 打开图片 -> RGB 像素 -> NumPy 数组
```

如果图片是 4x4 的 RGB 图，读出来的 shape 仍然是：

```text
(4, 4, 3)
```

所以 JSON 里的三层 list 和 PNG 图片文件只是两种不同的保存方式。进入模型前，它们都要变成数组或 tensor。

## 图像为什么可以看成三维数组

一张彩色图片由很多像素组成。每个像素通常有 3 个颜色通道：

- R：红色强度；
- G：绿色强度；
- B：蓝色强度。

如果一张图高度是 4、宽度是 4，每个像素有 3 个通道，那么它的 shape 就是：

```text
(4, 4, 3)
```

含义是：

```text
高度 H = 4
宽度 W = 4
通道 C = 3
```

例如 `[255, 0, 0]` 表示红色，因为红色通道最大，绿色和蓝色都是 0。`[255, 255, 255]` 表示白色，`[0, 0, 0]` 表示黑色。

## HWC 和 CHW 的区别

常见图片文件和很多图像处理库喜欢用 HWC：

```text
HWC = Height, Width, Channel
```

也就是：

```text
(高度, 宽度, 通道)
```

例如：

```text
(4, 4, 3)
```

但很多深度学习框架，尤其是 PyTorch，常用 CHW：

```text
CHW = Channel, Height, Width
```

也就是：

```text
(通道, 高度, 宽度)
```

例如：

```text
(3, 4, 4)
```

所以代码里经常会看到：

```python
arr_chw = arr_norm.transpose(2, 0, 1)
```

它的意思是把原来的第 2 维通道挪到最前面。

## 为什么深度学习模型通常需要归一化

图片原始像素值通常在 0 到 255 之间。模型训练和推理时，通常更喜欢范围稳定的浮点数，比如 0 到 1，或者进一步标准化到接近 0 均值。

今天用的是最基础的归一化：

```python
arr_norm = arr / 255.0
```

这样：

- 0 会变成 0.0；
- 128 会接近 0.502；
- 255 会变成 1.0。

归一化的工程意义是让输入数值范围更稳定。模型优化时，如果输入范围忽大忽小，训练会更难；如果输入范围统一，模型更容易学习。

## 和 CLIP 的关系

CLIP 需要同时处理图像和文本。图像进入 CLIP 之前，会被 resize、crop、转成 tensor、归一化，然后送进 image encoder。

所以你以后看到 CLIP 预处理时，不要觉得神秘。本质上就是：

```text
图片文件 -> 数组 -> 归一化 tensor -> 图像编码器
```

## 和 Stable Diffusion / VAE 的关系

Stable Diffusion 不是直接一直在像素空间里生成大图，而是常通过 VAE 把图像压缩到 latent space。

可以先这样理解：

```text
RGB 图片数组 -> VAE encoder -> latent 数组
latent 数组 -> VAE decoder -> RGB 图片数组
```

今天学的是 RGB 数组的最小直觉。后面学 VAE 时，你会看到 latent 也是数组，只是它的通道数、空间尺寸和含义不再是普通 RGB。

## 和 DiT 的关系

DiT 是 Diffusion Transformer。Transformer 通常处理 token，所以图像会被切成 patch，再变成一串 token。

可以先这样理解：

```text
图像数组 -> 切成小块 patch -> 每个 patch 变成 token -> Transformer 处理 token
```

所以 shape 很重要。你必须知道当前数据是：

- 一张图；
- 一批图；
- HWC；
- CHW；
- patch 序列；
- latent 特征图。

模型报错时，很多问题其实都和 shape 不匹配有关。
