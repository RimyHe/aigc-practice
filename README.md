# AIGC / 多模态算法每日实践

这个仓库是我的 AIGC / 多模态算法实习准备作品集实践区。它不追求一次做一个很大的项目，而是每天完成一个 30 到 60 分钟的小练习：跑通、读懂、修改关键片段，并把过程沉淀成可以回看的代码、样例数据、日志和中文说明。

## 项目定位

另一个“实习准备”项目负责每天的知识讲解、任务说明、资讯和面试表达；这个仓库专门负责把每日实践任务落地成真实文件。

每个练习尽量回答 5 个问题：

1. 输入是什么；
2. 中间处理了什么；
3. 输出是什么；
4. 这段代码为什么这样写；
5. 它和 AIGC / 多模态 / 图像生成实习有什么关系。

## 学习方式

每一天一个文件夹，例如：

- `day01_python_json`
- `day04_numpy_image_array`
- `day06_torch_linear_regression_loop`

每个文件夹尽量包含：

- `README.md`：目标、运行方式、整体思路、逐行代码解释、重点理解、自查题和参考答案。
- 样例数据：如 `json`、`csv`、`txt`、小图片、实验记录等。
- 可运行脚本：代码控制在适合阅读和修改的规模。
- `outputs/`：保存脚本运行结果、日志或生成文件。
- `notes.md`：总结关键概念、常见错误、工程联系、面试表达。

## 当前进度

- `day01_python_json`：Python 读取 JSON，并筛选 AIGC / 多模态相关学习材料。
- `day02_path_project`：Python 路径处理、脚本结构、模块化和项目入口。
- `day03_csv_experiment_log`：Python 读写 CSV、AIGC 实验记录和简单统计。
- `day04_numpy_image_array`：NumPy 数组、shape、dtype、归一化和图像数据直觉。
- `day05_torch_tensor_autograd`：PyTorch Tensor、requires_grad、autograd 和最小优化过程。
- `day06_torch_linear_regression_loop`：PyTorch 线性回归、MSE loss、optimizer 和完整训练循环。

## 阶段规划

我的规划是先把工程和数学直觉补稳，再逐步接到 AIGC / 多模态模型。

第一阶段：Python 工程和数据处理

- 文件路径、项目入口、模块化；
- JSON / CSV / 实验日志；
- 可复现输出和简单统计。

第二阶段：NumPy、图像数组和 PyTorch 基础

- 图像是 HWC / CHW 数字数组；
- dtype、归一化、通道；
- Tensor、autograd、训练循环；
- 线性回归、loss、optimizer。

第三阶段：基础深度学习模型

- 多层感知机；
- CNN 和图像分类；
- 数据集、DataLoader、训练/验证拆分；
- 保存模型和实验记录。

第四阶段：多模态和图像生成核心

- embedding 和相似度；
- CLIP 图文对齐；
- VAE 和 latent space；
- U-Net、扩散模型、Stable Diffusion 推理流程；
- DiT、flow matching、MoE、Sparse DiT 等进阶方向。

第五阶段：AIGC 工程落地

- API demo；
- 图像编辑 workflow；
- prompt / seed / guidance scale / steps 实验记录；
- 小型作品集 demo 和面试表达整理。

## README 写作规范

之后每天的练习 `README.md` 默认按下面的要求写：

1. `整体思路` 要具体到操作顺序和运行指令，不只写抽象流程。
2. `逐行解释代码` 要像老师带读一样，把主脚本按代码块拆开，解释每句在做什么、为什么需要它、和输入输出流程有什么关系。
3. `重点理解` 不只列知识点，要解释为什么这样写，并尽量联系算法工程、AIGC、多模态或图像生成场景。
4. `自查清单` 要配参考答案，方便自己对照。
5. `可修改的小练习` 要给对应的指引和参考解法，避免卡在“知道要改但不知道怎么改”。

## 每日练习流程

1. 先打开当天文件夹的 `README.md`。
2. 按说明运行最小脚本，确认能看到输出。
3. 对照样例数据看清楚输入结构。
4. 阅读逐行代码解释，理解每个关键变量和操作。
5. 修改 2 到 3 个小地方，再运行观察变化。
6. 用自己的话总结今天最重要的 3 个点。

## Git 分工

为了练习 Git，同时保持 GitHub 首页更新，之后采用这个分工：

我负责维护并推送项目总 README，也就是这个文件。它作为作品集首页，记录当前进度、规划和学习路线。

我提交总 README 时使用：

```powershell
git add README.md
git commit -m "Update practice index"
git push
```

如果模板也更新，再提交：

```powershell
git add README.md PRACTICE_TEMPLATE.md
git commit -m "Update practice README template"
git push
```

当天练习文件夹由我自己练习提交，例如：

```powershell
git add day06_torch_linear_regression_loop
git commit -m "Add day06 torch linear regression practice"
git push
```

这样总目录和每日练习可以分开管理：总 README 保持 GitHub 首页可读，每天的项目文件夹则用于练习 Git add / commit / push。
