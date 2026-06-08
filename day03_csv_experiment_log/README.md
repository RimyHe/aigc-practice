# Day 03 - CSV 实验记录和简单统计

今天的目标是把前两天的 prompt 和路径基础，升级成一个“实验记录小工具”。它模拟 AIGC / 图像生成实验里常见的流程：记录每次生成的参数和结果，再用脚本统计平均分、最佳实验和不同模型的实验数量。

## 今日目标

完成后你应该能说清楚：

1. 为什么 AIGC 实验不能只保存图片，还要记录 prompt、seed、steps 等参数；
2. 如何用 `csv.DictReader` 把 CSV 表格读成一条条字典；
3. 为什么从 CSV 读出来的数字一开始是字符串，必须手动转成 `int` 或 `float`；
4. 如何按 `model` 字段做简单分组统计；
5. 如何把统计结果同时打印到终端并保存到 `outputs/summary.txt`。

## 文件结构

```text
day03_csv_experiment_log/
  README.md
  data/
    generation_runs.csv
  outputs/
    summary.txt
  summarize_runs.py
  notes.md
```

- `README.md`：今天的练习说明。
- `data/generation_runs.csv`：9 条模拟生成实验记录。
- `outputs/summary.txt`：运行脚本后生成的统计结果。
- `summarize_runs.py`：主脚本，读取 CSV、转换数字、统计结果、保存总结。
- `notes.md`：解释 AIGC 实验为什么要记录这些字段。

## 先打开哪个文件

建议按这个顺序看：

1. 先打开 `data/generation_runs.csv`，看输入表格有哪些字段。
2. 再打开 `summarize_runs.py`，重点看 `main()`、`load_runs()`、`build_summary()`。
3. 然后运行脚本，观察终端输出。
4. 最后打开 `outputs/summary.txt`，确认统计结果已经保存。

可以用：

```powershell
cd D:\1AIGC_daily_practice\day03_csv_experiment_log
notepad data\generation_runs.csv
notepad summarize_runs.py
```

## 运行命令

进入当天文件夹：

```powershell
cd D:\1AIGC_daily_practice\day03_csv_experiment_log
```

运行脚本：

```powershell
python summarize_runs.py
```

运行后打开输出文件：

```powershell
notepad outputs\summary.txt
```

## 应该观察什么输出

终端里应该看到类似：

```text
AIGC 生成实验统计结果
============================
总实验数：9
平均 CLIP 分数：0.304
平均人工评分：4.09

人工评分最高的实验：
- run_id：r008
- model：Image Editing
- prompt：Turn a sketch of a chair into a realistic product render
...

按模型统计实验数量：
- Stable Diffusion：4 次
- DiT：3 次
- Image Editing：2 次
```

你要重点观察三件事：

1. 总实验数是否等于 CSV 里的数据行数；
2. 人工评分最高的实验是否对应 `r008`；
3. `outputs/summary.txt` 的内容是否和终端输出一致。

## 整体思路

这个练习模拟的是一次最小实验复盘：

1. 进入练习文件夹：

```powershell
cd D:\1AIGC_daily_practice\day03_csv_experiment_log
```

2. 查看 CSV 输入：

```powershell
notepad data\generation_runs.csv
```

每一行代表一次生成实验。比如：

```text
r001,"A cyberpunk street after rain, neon reflections",Stable Diffusion,42,7.5,30,0.312,4.2,"good color and composition"
```

它表示：第 `r001` 次实验，用 Stable Diffusion，在 seed 为 42、guidance scale 为 7.5、steps 为 30 的设置下生成一张图，CLIP 分数是 0.312，人工评分是 4.2。

3. 运行统计脚本：

```powershell
python summarize_runs.py
```

4. 脚本内部按这个顺序工作：

- 用 `Path(__file__).parent` 找到当前脚本所在目录；
- 拼出 `data/generation_runs.csv` 和 `outputs/summary.txt`；
- 用 `csv.DictReader` 读取 CSV；
- 把 `seed`、`steps` 转成整数；
- 把 `guidance_scale`、`clip_score`、`human_rating` 转成浮点数；
- 统计总实验数、平均 CLIP 分数、平均人工评分；
- 找到人工评分最高的一条实验；
- 按 `model` 统计每个模型的实验数量；
- 把结果打印出来，并保存到 `outputs/summary.txt`。

5. 打开输出文件：

```powershell
notepad outputs\summary.txt
```

这一步很重要。真实工程里，脚本运行完不能只看终端，因为终端内容会消失；把结果保存到文件，才方便后续写报告、做对比、提交作品集。

## 重点理解的 3 个代码点

### 1. `csv.DictReader`

代码：

```python
reader = csv.DictReader(f)
```

`DictReader` 会把 CSV 的每一行读成一个字典。表头会变成字典的 key，比如 `run_id`、`prompt`、`model`、`clip_score`。

这样你就可以写：

```python
row["clip_score"]
row["human_rating"]
row["model"]
```

而不是用第 0 列、第 1 列、第 2 列这种不直观的方式。

工程联系：AIGC 实验记录往往有很多字段，用字段名访问比用列号安全得多。以后字段顺序调整了，只要表头名字不变，代码仍然容易维护。

### 2. 字符串转数字

代码：

```python
row["clip_score"] = float(row["clip_score"])
row["human_rating"] = float(row["human_rating"])
```

CSV 文件本质上是文本，所以读出来的所有内容一开始都是字符串。比如 `0.312` 看起来是数字，但 Python 读出来可能是 `"0.312"`。

如果不转成数字，就不能正确做平均值、排序、比较大小。

工程联系：很多实验数据来自 CSV、JSON、日志文件或网页接口。读进来后第一步经常是类型转换。类型不对时，模型代码不一定马上报明显错误，但统计结果会错。

### 3. 按字段分组统计

代码：

```python
model_counts = {}
for run in runs:
    model = run["model"]
    model_counts[model] = model_counts.get(model, 0) + 1
```

这段代码按 `model` 分组，统计每个模型出现了多少次。

工程联系：真实实验里经常要按模型、prompt 类型、分辨率、参数区间、数据集来源做分组。比如你可能想知道 Stable Diffusion 和 DiT 哪个平均人工评分更高，或者 image editing 任务在哪个 guidance scale 下更稳定。

## 常见错误

### 1. CSV 表头拼错

如果把 `clip_score` 写成 `clipscore`，代码里的：

```python
row["clip_score"]
```

就会找不到这个字段。

参考检查：确保表头仍然是：

```text
run_id,prompt,model,seed,guidance_scale,steps,clip_score,human_rating,notes
```

### 2. 数字字段没转类型

如果忘记 `float()`，平均分会算不了，或者比较大小时逻辑不可靠。

参考答案：`clip_score` 和 `human_rating` 应该转成 `float`，`seed` 和 `steps` 应该转成 `int`。

### 3. 路径找不到

如果报 `FileNotFoundError`，先检查：

- 你是否还在 `D:\1AIGC_daily_practice\day03_csv_experiment_log`；
- `data/generation_runs.csv` 是否存在；
- 代码里的文件名有没有拼错。

这里脚本使用 `Path(__file__).parent`，所以即使你从项目根目录运行，也仍然能找到数据文件。

### 4. `outputs` 不存在

脚本里已经写了：

```python
output_path.parent.mkdir(parents=True, exist_ok=True)
```

所以即使你删除 `outputs/`，再次运行脚本也会自动创建。

## 自查清单和参考答案

- 问：CSV 读出来的每一行是什么结构？
  答：每一行是一个 dict，key 来自 CSV 表头，value 是当前行对应字段的文本值。

- 问：为什么要把 `clip_score` 转成 `float`？
  答：因为要计算平均值和比较大小。CSV 读出来的 `clip_score` 本来是字符串，必须转成浮点数才能做数值计算。

- 问：人工评分最高的是哪条实验？
  答：默认数据里是 `r008`，它的 `human_rating` 是 4.7。

- 问：Stable Diffusion 有几条实验？
  答：默认数据里有 4 条，分别是 `r001`、`r002`、`r003`、`r007`。

- 问：这个练习和 AIGC 实习有什么关系？
  答：AIGC 实验需要反复比较模型、prompt、seed、steps 和评分结果。这个练习就是最小版的实验记录、统计和结果保存流程。

## 可修改的小练习

### 练习 1：新增一条实验记录

目标：练习修改 CSV 输入，并观察统计结果变化。

指引：打开 `data/generation_runs.csv`，在最后新增一行：

```text
r010,"A cinematic portrait of a robot scientist",Stable Diffusion,808,7.0,30,0.325,4.3,"good face detail and lighting"
```

参考结果：再次运行后，总实验数应该从 9 变成 10，Stable Diffusion 的数量应该从 4 变成 5。

### 练习 2：增加按 `guidance_scale` 排序输出

目标：练习在已有数据上做排序。

指引：在 `build_summary()` 里 `model_counts = count_by_model(runs)` 后面加入：

```python
highest_guidance_run = max(runs, key=lambda run: run["guidance_scale"])
```

然后在 `lines` 里增加：

```python
"",
"guidance_scale 最高的实验：",
f"- run_id：{highest_guidance_run['run_id']}",
f"- guidance_scale：{highest_guidance_run['guidance_scale']}",
```

参考结果：默认数据里 guidance scale 最高的是 `r008`，数值是 8.5。

### 练习 3：筛选 CLIP 分数大于 0.30 的结果

目标：练习用条件筛选实验记录。

指引：在 `build_summary()` 里增加：

```python
high_clip_runs = [run for run in runs if run["clip_score"] > 0.30]
```

然后在 `lines` 里增加：

```python
"",
f"CLIP 分数大于 0.30 的实验数：{len(high_clip_runs)}",
```

参考结果：默认数据里应该有 5 条实验的 CLIP 分数大于 0.30：`r001`、`r003`、`r005`、`r007`、`r008`。

## 面试表达

可以这样总结今天的练习：

这个练习模拟了 AIGC 生成实验的记录和复盘流程。我用 CSV 保存每次生成的 prompt、模型、seed、guidance scale、steps、CLIP score 和人工评分，再用 Python 标准库读取并统计平均指标、最佳实验和模型分布。它对应真实项目里实验管理、结果分析和报告整理的基础能力。
