# Day 01 - Python JSON 素材筛选小实验

今天的目标不是从零写完整项目，而是先跑通一个很小的 AIGC 学习资料筛选脚本，理解 Python 在算法实习里常见的工作方式：

1. 读取结构化数据；
2. 遍历每一条记录；
3. 根据关键词筛选；
4. 把结果打印成清单。

## 文件说明

- `materials.json`：已经准备好的样例学习材料。
- `filter_materials.py`：主脚本，读取 JSON 并筛选图像生成 / 多模态相关材料。

## 运行方式

在这个文件夹下运行：

```powershell
cd D:\1AIGC_daily_practice\day01_python_json
python filter_materials.py
```

如果你在项目根目录，也可以运行：

```powershell
cd D:\1AIGC_daily_practice
python .\day01_python_json\filter_materials.py
```

## 整体思路

这个练习模拟的是算法实习里很常见的一类小任务：有一批结构化资料或样本，你需要按规则筛出和当前方向相关的内容。真实工程里，这类数据可能是论文清单、图片路径、prompt 列表、训练样本元信息、实验记录或评测结果。

具体执行顺序如下：

1. 进入当天练习文件夹：

```powershell
cd D:\1AIGC_daily_practice\day01_python_json
```

2. 先打开样例数据：

```powershell
notepad materials.json
```

观察每条资料都有 4 个字段：

- `title`：资料标题。
- `direction`：资料方向。
- `keywords`：关键词列表。
- `note`：为什么这条资料有用。

3. 再打开脚本：

```powershell
notepad filter_materials.py
```

先不用急着逐行看，先抓住主流程：脚本读取 `materials.json`，把里面的 JSON 数据变成 Python 对象，然后逐条检查 `keywords` 是否包含 AIGC / 多模态相关关键词，最后把匹配到的资料打印出来。

4. 运行脚本：

```powershell
python filter_materials.py
```

你应该看到 3 条结果：CLIP、Stable Diffusion、Diffusion Transformer。没有出现 Python 文件处理和 Pandas 表格统计，是因为它们的关键词没有和脚本里的 `KEYWORDS` 产生交集。

5. 对照输入和输出：

- `CLIP` 这一条包含 `CLIP` 和 `multimodal`，所以会被选中。
- `Stable Diffusion` 这一条包含 `diffusion`，所以会被选中。
- `Diffusion Transformer` 这一条包含 `diffusion` 和 `image generation`，所以会被选中。
- `Pandas 表格统计基础` 这一条只有 `pandas`、`csv`、`statistics`，所以不会被选中。

6. 如果运行失败，先按这个顺序检查：

- 终端是否已经进入 `D:\1AIGC_daily_practice\day01_python_json`。
- 当前文件夹里是否能看到 `filter_materials.py` 和 `materials.json`。
- `materials.json` 里有没有漏掉逗号、引号或中括号。
- 电脑里是否能用 `python` 命令启动 Python。

## 今天重点理解

重点看 `filter_materials.py` 里的这三处：

1. `KEYWORDS`：为什么用集合保存关键词。

   代码里写的是：

   ```python
   KEYWORDS = {"diffusion", "CLIP", "image generation", "multimodal"}
   ```

   集合适合做“有没有某个元素”“两个集合有没有共同元素”这类判断。工程里经常会遇到类似需求：筛选某一类样本、判断一个图片是否带有某个标签、判断一个 prompt 是否属于某个任务类型。用集合可以让判断逻辑更清楚，不需要写很多个 `if keyword == ...`。

2. `json.load(f)`：如何把 JSON 文件变成 Python 里的 list / dict。

   `materials.json` 在文件里是文本，但程序不能直接拿文本做结构化处理。`json.load(f)` 会把它变成 Python 对象：最外层是一个 `list`，里面每个元素是一个 `dict`。这和很多 AIGC 工程数据很像，比如一个训练数据集清单通常也是“很多条样本”，每条样本里有图片路径、文本描述、标签、来源等字段。

3. `tags & KEYWORDS`：为什么这一句能判断两个集合有没有交集。

   代码里先把当前资料的关键词变成集合：

   ```python
   tags = set(item["keywords"])
   ```

   然后用：

   ```python
   return bool(tags & KEYWORDS)
   ```

   `tags & KEYWORDS` 表示两个集合的交集。如果交集不为空，说明这条资料至少有一个关键词命中了目标方向。工程上这就是一个最小版的规则过滤器：先用标签或元信息做粗筛，再把相关样本交给后续模型、评测脚本或人工分析。

## 你可以做的小修改

1. 给 Pandas 那条资料增加一个相关关键词。

   指引：打开 `materials.json`，找到 `Pandas 表格统计基础`，把它的 `keywords` 改成：

   ```json
   ["pandas", "csv", "statistics", "multimodal"]
   ```

   参考结果：再次运行后，Pandas 这一条也会被打印出来。原因是 `multimodal` 在 `KEYWORDS` 里。

2. 修改脚本里的目标关键词。

   指引：打开 `filter_materials.py`，把：

   ```python
   KEYWORDS = {"diffusion", "CLIP", "image generation", "multimodal"}
   ```

   改成：

   ```python
   KEYWORDS = {"csv", "statistics"}
   ```

   参考结果：再次运行后，应该主要筛出 `Pandas 表格统计基础`。这说明筛选规则是由 `KEYWORDS` 控制的。

3. 新增一条 VAE 资料，并让它能被筛出来。

   指引：先在 `materials.json` 里新增一条：

   ```json
   {
     "title": "VAE 编码和解码直觉",
     "direction": "图像生成算法",
     "keywords": ["VAE", "latent space", "image generation"],
     "note": "VAE 帮助理解 Stable Diffusion 里图像和 latent 表示之间的转换。"
   }
   ```

   注意：JSON 列表里每两个对象之间要有逗号。

   参考结果：如果 `KEYWORDS` 里保留了 `image generation`，这条 VAE 资料会被筛出来；如果你想专门练 VAE，也可以把 `VAE` 加进 `KEYWORDS`。

## 自查和参考答案

- 问：`materials` 是 list 还是 dict？
  答：`materials` 是 list。因为 `materials.json` 最外层是 `[...]`，表示一个列表，列表里放着多条资料。

- 问：每个 `item` 是什么结构？
  答：每个 `item` 是 dict。它包含 `title`、`direction`、`keywords`、`note` 这几个键。可以把它理解成一条资料的元信息。

- 问：`item["keywords"]` 是什么？
  答：它是一个 list，例如 `["CLIP", "multimodal", "image-text alignment"]`。脚本会把这个 list 转成 set，方便和 `KEYWORDS` 做交集判断。

- 问：`tags & KEYWORDS` 的含义是什么？
  答：它表示当前资料关键词集合和目标关键词集合的交集。只要交集不是空集合，就说明这条资料和目标方向相关。

- 问：为什么输出里有 CLIP、Stable Diffusion、Diffusion Transformer？
  答：因为它们至少有一个关键词命中了 `KEYWORDS`。CLIP 命中 `CLIP` 或 `multimodal`，Stable Diffusion 命中 `diffusion`，Diffusion Transformer 命中 `diffusion` 或 `image generation`。

- 问：这个练习和 AIGC / 多模态实习有什么关系？
  答：实习里经常需要处理结构化数据，比如图片路径清单、prompt 数据集、模型评测结果、论文资料列表。这个练习就是最小版的数据读取和规则筛选流程，后面换成图片、文本、embedding 或实验结果，本质流程仍然类似。
