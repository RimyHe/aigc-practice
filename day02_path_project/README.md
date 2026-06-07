# Day 02 - Python 路径处理和项目入口

今天的目标是把 Day 01 的 JSON 筛选脚本升级成一个更像真实 AIGC workflow 的小项目。你不需要从零写完整代码，重点是跑通、读懂、修改关键片段，并理解一个小项目里数据、脚本和输出应该如何组织。

## 今日目标

完成后你应该能说清楚：

1. 数据文件应该放在哪里，脚本如何稳定找到它；
2. 输出文件应该放在哪里，代码如何自动创建输出目录；
3. 为什么真实项目里常写 `main()`，而不是把所有逻辑直接写在文件顶层；
4. 如何把读取、筛选、保存拆成清楚的小函数。

## 文件结构

```text
day02_path_project/
  README.md
  data/
    prompts.json
  outputs/
  main.py
```

- `README.md`：今天的练习说明。
- `data/prompts.json`：5 条样例 prompt，每条都有 `id`、`prompt`、`tags`。
- `outputs/`：保存筛选结果的目录。
- `main.py`：主脚本，负责读取 prompt、筛选 AIGC / 多模态相关条目，并保存结果。

## 先打开哪个文件

建议按这个顺序打开：

1. 先打开 `data/prompts.json`，看清楚输入数据长什么样。
2. 再打开 `main.py`，先看 `main()` 里的主流程。
3. 最后看 `outputs/selected_prompts.json`，它会在运行脚本后自动生成。

可以用：

```powershell
cd D:\1AIGC_daily_practice\day02_path_project
notepad data\prompts.json
notepad main.py
```

## 运行命令

进入当天文件夹：

```powershell
cd D:\1AIGC_daily_practice\day02_path_project
```

运行脚本：

```powershell
python main.py
```

运行后再打开输出文件：

```powershell
notepad outputs\selected_prompts.json
```

## 应该观察什么输出

终端里应该看到类似：

```text
读取 prompt 总数：5
筛选命中数量：4
输出文件：D:\1AIGC_daily_practice\day02_path_project\outputs\selected_prompts.json
```

为什么是 4 条？因为 5 条 prompt 里，除了 `classification` 那条主要是普通分类任务，其余 prompt 至少命中了下面目标标签之一：

- `diffusion`
- `CLIP`
- `DiT`
- `multimodal`
- `text-to-image`
- `image-editing`

输出文件 `outputs/selected_prompts.json` 里应该只保存被筛选出来的 prompt。

## 整体思路

这个练习模拟的是一个最小 AIGC 数据准备流程：原始 prompt 放在 `data/`，脚本读取它，按标签筛选出与 AIGC / 多模态相关的 prompt，再把结果保存到 `outputs/`。

具体执行流程：

1. 进入项目目录：

```powershell
cd D:\1AIGC_daily_practice\day02_path_project
```

2. 观察输入数据：

```powershell
notepad data\prompts.json
```

每条数据像这样：

```json
{
  "id": "p001",
  "prompt": "A tiny robot painting a watercolor city at sunrise",
  "tags": ["text-to-image", "diffusion", "image generation"]
}
```

你要注意三件事：`id` 用来唯一标记样本，`prompt` 是文本输入，`tags` 是后续筛选的依据。真实图像生成或多模态项目里，也经常会有类似的样本清单，只是还会多出图片路径、分辨率、数据来源、人工评分等字段。

3. 运行脚本：

```powershell
python main.py
```

4. 脚本内部按这个顺序工作：

- 用 `Path(__file__).parent` 找到 `main.py` 所在目录；
- 拼出输入路径 `data/prompts.json`；
- 拼出输出路径 `outputs/selected_prompts.json`；
- 用 `json.load` 读取 prompt 列表；
- 用 `AIGC_TAGS` 和每条数据的 `tags` 做交集判断；
- 打印读取总数和筛选数量；
- 如果 `outputs` 不存在，就自动创建；
- 用 `json.dump` 保存筛选结果。

5. 打开输出文件：

```powershell
notepad outputs\selected_prompts.json
```

把输出文件和原始 `data/prompts.json` 对照看：被保留下来的条目，都至少有一个标签命中了 `AIGC_TAGS`。

## 重点理解的 3 个代码点

### 1. `Path(__file__).parent`

代码：

```python
project_dir = Path(__file__).parent
input_path = project_dir / "data" / "prompts.json"
output_path = project_dir / "outputs" / "selected_prompts.json"
```

`__file__` 表示当前脚本文件的位置，`Path(__file__).parent` 表示当前脚本所在文件夹。这样写的好处是：脚本不依赖你从哪个目录运行它，只要文件结构不变，它就能找到自己的 `data/` 和 `outputs/`。

工程联系：AIGC 项目里经常有 `data/`、`configs/`、`checkpoints/`、`outputs/` 这些目录。如果路径写死成某个临时位置，换电脑、换文件夹、换终端运行位置就容易报错。用 `Path` 拼路径，是为了让脚本更稳定、更容易迁移。

### 2. `main()`

代码：

```python
def main():
    project_dir = Path(__file__).parent
    input_path = project_dir / "data" / "prompts.json"
    output_path = project_dir / "outputs" / "selected_prompts.json"

    prompts = load_prompts(input_path)
    selected_prompts = select_prompts(prompts)
    save_prompts(selected_prompts, output_path)
```

`main()` 把脚本主流程集中起来：先准备路径，再读取数据，再筛选，再保存。你看 `main()` 就能知道整个程序做了什么。

工程联系：真实算法工程里，主流程通常会更长，比如读取配置、加载模型、读取数据、推理、评测、保存结果。如果没有 `main()`，代码容易散在文件顶层，后面想复用函数、写测试、被其他脚本导入时都会变乱。

### 3. `json.dump(... ensure_ascii=False, indent=2)`

代码：

```python
json.dump(prompts, f, ensure_ascii=False, indent=2)
```

`json.dump` 负责把 Python 对象写回 JSON 文件。`ensure_ascii=False` 的作用是保留中文本身，不把中文转成一串 `\u4e2d` 这样的编码。`indent=2` 的作用是让输出文件有缩进，方便人阅读。

工程联系：实验结果、prompt 清单、评测记录经常要保存成 JSON。如果保存出来的文件很难读，就不利于检查数据是否正确。算法工程里很多错误不是模型本身的问题，而是数据或配置文件在某一步被写错了，所以输出文件的可读性很重要。

## 常见错误

### 1. `FileNotFoundError`

可能原因：脚本找不到 `data/prompts.json`。

参考检查：

- 确认你没有改动 `data` 文件夹名字；
- 确认 `prompts.json` 文件还在 `data/` 里面；
- 确认代码里路径仍然是 `project_dir / "data" / "prompts.json"`。

### 2. JSON 格式错误

可能原因：手动修改 `prompts.json` 时漏了逗号、双引号或中括号。

参考检查：

- JSON 字符串必须用双引号；
- 每两个对象之间要有逗号；
- 最外层是 `[` 和 `]`；
- 最后一条数据后面不要多写逗号。

### 3. 中文乱码

可能原因：读取或保存文件时没有指定 UTF-8。

参考检查：

- 读取时使用 `encoding="utf-8"`；
- 保存时也使用 `encoding="utf-8"`；
- `json.dump` 里保留 `ensure_ascii=False`。

### 4. `outputs` 不存在

这个脚本已经处理了这个问题：

```python
output_path.parent.mkdir(parents=True, exist_ok=True)
```

如果你手动删除 `outputs/`，再次运行脚本时它会自动创建。

## 自查清单和参考答案

- 问：为什么不直接写 `"data/prompts.json"`？
  答：直接写相对路径会依赖终端当前所在目录。使用 `Path(__file__).parent` 后，脚本会从自己所在的位置找 `data/`，更稳定。

- 问：`main()` 里最重要的流程是什么？
  答：准备路径、读取输入、筛选数据、保存输出、打印结果。这就是很多数据处理脚本的基本骨架。

- 问：筛选逻辑在哪里？
  答：在 `is_aigc_related()` 里。它把当前 prompt 的 `tags` 转成集合，再和 `AIGC_TAGS` 做交集判断。

- 问：为什么 `classification` 那条没有被筛出来？
  答：它的标签是 `classification` 和 `vision`，没有命中 `diffusion`、`CLIP`、`DiT`、`multimodal`、`text-to-image`、`image-editing`。

- 问：输出文件为什么中文不会变成 `\uXXXX`？
  答：因为保存时使用了 `ensure_ascii=False`，它会保留中文字符本身。

## 可修改的小练习

### 练习 1：新增一条 DiT prompt

目标：练习修改输入数据，并确认脚本能自动筛选。

指引：打开 `data/prompts.json`，在最后一条数据后面新增一条。注意上一条后面要加逗号。

参考新增内容：

```json
{
  "id": "p006",
  "prompt": "A futuristic city generated by a sparse Diffusion Transformer",
  "tags": ["DiT", "diffusion", "text-to-image"]
}
```

参考结果：再次运行 `python main.py` 后，读取总数应该变成 6，筛选命中数量应该变成 5。

### 练习 2：修改关键词集合

目标：理解筛选结果由 `AIGC_TAGS` 控制。

指引：打开 `main.py`，把：

```python
AIGC_TAGS = {
    "diffusion",
    "CLIP",
    "DiT",
    "multimodal",
    "text-to-image",
    "image-editing",
}
```

临时改成：

```python
AIGC_TAGS = {"classification"}
```

参考结果：再次运行后，筛选命中数量应该变成 1，输出文件里主要会保留分类任务那条。

### 练习 3：增加标签统计输出

目标：练习在已有主流程上加一个小功能。

指引：先在 `main()` 里 `print(f"筛选命中数量：{len(selected_prompts)}")` 后面增加：

```python
for item in selected_prompts:
    print(item["id"], item["tags"])
```

参考结果：终端会多打印每条被选中 prompt 的 `id` 和 `tags`。这在工程里很常见：先不要急着画图或训练模型，而是把关键中间结果打印出来，确认数据确实是你想要的。

## 面试表达

可以这样总结今天的练习：

这个练习把一个简单 JSON 筛选脚本整理成了更像项目的结构。我用 `Path(__file__).parent` 稳定定位数据目录，用 `main()` 组织读取、筛选和保存流程，并用 `json.dump` 把结果保存到输出目录。它对应 AIGC 工程里常见的 prompt 数据准备、样本筛选和实验结果落盘流程。
