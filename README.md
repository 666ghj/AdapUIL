## UILModel-XXQ 运行指南（中文）

本项目实现了基于重启随机游走（RWR）与多种亲和矩阵策略的对齐表示学习流程：
- 读取跨网络图数据与锚节点对（groundtruth）
- 在源图/目标图/融合图上进行随机游走，生成序列
- 以 Word2Vec 训练节点嵌入
- 通过 MLP 进行源-目标嵌入映射并评估对齐效果

默认数据集为 `dataspace/douban`，默认方案为 `solution=23`（两网融合游走 + 亲和矩阵 + 同时重启，序列一阶起始）。


### 目录结构与关键文件

```
UILModel-XXQ/
  CPython_GeneratingSequences/     # C 混编的亲和矩阵游走实现（Windows .dll / Linux .so）
  dataspace/                       # 内置数据集（douban、Facebook/Twitter）
  models/                          # 训练出的 Word2Vec 模型存放目录（示例：douban）
  RWR_UIL/                         # RWR 相关实现（游走与数据组装）
  source/                          # Word2Vec、映射模型与评估工具
  CudaTest.py                      # CUDA 是否可用的快速自检
  RWR_UIL.py                       # 项目主入口（训练 + 映射 + 评估）
  parameter_setting.py             # 全局参数统一配置（alpha、emb_size、solution 等）
  Experimental control.py          # 批量实验脚本（会动态改写 parameter_setting.py）
```

- `RWR_UIL.py`：主入口。通过 `RWR_get_stg()` 读取数据，然后分别调用 `RWR_source()` 与 `RWR_target()` 生成向量，接着用 `PaleMappingMlp` 做映射训练与评估，最后打印 Accuracy/MAP/AUC/Hit/Precision@k。
- `parameter_setting.py`：统一参数配置（详见下文“参数说明”）。
- `RWR_UIL/RWR_get_stg.py`：数据装载与可选的网络结构增强（如三元闭包、删叶子节点等，默认关闭）。在两网融合场景下对目标图节点做偏移（默认写死偏移为 4096）。
- `RWR_UIL/RWR_DeepWalk.py`：核心游走实现，支持多种方案：
  - 纯 RWR 随机游走
  - 基于亲和矩阵的游走（两种亲和矩阵累加策略）
  - 两网融合后的游走（含锚节点跳转）
  - C 动态库加速（Windows 使用 .dll，Linux 使用 .so）
  - 组合/加权/拼接等多种 `solution` 方案
- `source/deepwalk.py`：标准 DeepWalk（gensim）实现，项目主流程默认走 `RWR_DeepWalk`。
- `source/mapping_model.py`：源→目标映射模型（线性/两层 MLP），默认使用 `PaleMappingMlp`。
- `source/evaluate.py`：评估指标（Accuracy、MAP、AUC、Hit、Precision@k 等）。
- `source/NetWork_Opt.py`：可选的网络结构优化（网络增强、三元闭包、删叶子节点）。
- `CPython_GeneratingSequences/*.dll|*.so`：C 实现的高效亲和矩阵游走。


### 环境准备（Windows/PowerShell）

这个环境很好配，缺什么包装什么就好了。


### 数据准备

默认使用 `dataspace/douban`：
- `groundtruth`：两列锚节点对（source_id target_id），用于监督映射与评估
- `online.edgelist`：源图边列表（空格分隔的整型节点对）
- `offline.edgelist`：目标图边列表
- `sourece_neg_douban.txt`、`target_neg_douban.txt`：负样本边

切换为 Facebook/Twitter 数据，请参考“切换数据集”。


### 快速开始（默认 douban + solution=23）

```powershell
conda activate pytorch_python11
python RWR_UIL.py
```

控制台将打印：
- source/target 节点数
- 训练过程日志（映射训练 200 个 epoch 的 loss 进度简略打印）
- 评估指标：Accuracy、MAP、AUC、Hit、Precision@{5,10,15,20,25,30}
- 总时间（秒）


### 参数说明（`parameter_setting.py`）

```python
alpha = 0.4        # 重启概率（用于 RWR 与/或亲和矩阵权重）
emb_size = 256     # 嵌入维度
length_walk = 50   # 序列长度
num_walks = 50     # 每个节点游走轮数
window_size = 10   # Word2Vec 窗口
num_iters = 2      # Word2Vec 训练轮数
train_ratio = 0.8  # groundtruth 训练比例
val_ratio = 0.1    # groundtruth 验证比例

solution = 23      # 游走/组合方案编号（见下）

gt_tar_add5000 = None  # 锚节点数组（仅融合游走内部使用，默认由 RWR_get_stg 自动生成）
```

#### 关于 solution（部分常用方案速览）

- 0.1：`RWR_random_walk`（纯 RWR）
- 0.2：`RWR_random_walk_AffinityMatrix_efficient`（亲和矩阵 + C 加速）
- 0.3：`RWR_random_walk_union`（两网融合后游走，带锚节点跳转）
- 1：RWR + 亲和矩阵（alpha=0），1:1 拼接
- 2：RWR + 亲和矩阵（alpha 相同），1:1 拼接
- 3：RWR + 亲和矩阵（alpha 相同），向量加和
- 4/5/6：RWR + 亲和矩阵（alpha 相同），0.8:0.2 / 0.5:0.5 / 0.2:0.8 加权和
- 7/8：RWR + 亲和矩阵（第二种亲和矩阵累加策略），1:1 拼接
- 9：RWR + RWR，1:1 拼接
- 10/11/12：RWR + 亲和矩阵（不同 alpha），1:1 拼接
- 13/14/15/16：RWR 与 亲和矩阵不同比例维度拼接（总维度为 `emb_size`）
- 17/18/19/20/21/22/23：包含“融合游走”与“亲和矩阵 + 同时重启（C 加速）”等更复杂组合

默认 `solution=23`：`RWR_random_walk_union` 与 `RWR_random_walk_AffinityMatrix_with_restart2` 的 1:1 拼接。

提示：不同 `solution` 会影响游走策略（是否融合、是否用亲和矩阵、是否 C 加速、是否同时重启）与向量融合方式（拼接/加和/加权）。


### 切换数据集

打开 `RWR_UIL/RWR_get_stg.py`：

- Douban（默认启用）：
```python
gt = np.genfromtxt("dataspace/douban/groundtruth", dtype=np.int32)
get_src = np.genfromtxt("dataspace/douban/online.edgelist", dtype=np.int32)
get_tar = np.genfromtxt("dataspace/douban/offline.edgelist", dtype=np.int32)
src_neg_edges = np.genfromtxt("dataspace/douban/sourece_neg_douban.txt", dtype=np.int32)
tar_neg_edges = np.genfromtxt("dataspace/douban/target_neg_douban.txt", dtype=np.int32)
```

- Facebook/Twitter（将下列四行取消注释，同时注释掉 Douban 的四行）：
```python
get_src = np.genfromtxt("dataspace/Facebook/twitter.edges", dtype=np.int32)
get_tar = np.genfromtxt("dataspace/Facebook/facebook.edges", dtype=np.int32)
gt = np.genfromtxt("dataspace/Facebook/fb_tw.ground", dtype=np.int32)
src_neg_edges = np.genfromtxt("dataspace/Facebook/sourece_neg_tw.txt", dtype=np.int32)
tar_neg_edges = np.genfromtxt("dataspace/Facebook/target_neg_fb.txt", dtype=np.int32)
```

关于融合偏移：在融合游走里，目标图节点会整体偏移（在本代码中写死为 `4096`）。对应实现分别在：
- `RWR_UIL/RWR_get_stg.py`：`gt_tar_add5000[:, 1] += 4096` 与 `tar_add5000 = get_tar + 4096`
- `RWR_UIL/RWR_union.py`：取回偏移后的目标节点向量（索引从 `i+4096` 开始）


### 运行与结果

```powershell
conda activate pytorch_python11
python RWR_UIL.py
```

成功运行后，控制台会打印如下信息（示例）：
- 节点计数、游走/训练过程日志
- Accuracy、MAP、AUC、Hit、Precision@k
- 总耗时（秒）

嵌入结果默认不落盘（只在内存中继续用于映射与评估）。如需持久化，可在 `RWR_source.py` / `RWR_target.py` 中将 `vector_source/target` 写入文件。


### 批量实验（可选）

文件：`Experimental control.py`

- 会批量改写 `parameter_setting.py`（alpha/solution/emb_size）并连续运行 `python RWR_UIL.py`，将每次输出保存到 `1120实验/方法X/Y.txt`。
- 如需自定义方法或次数，修改 `methods` 列表与 `run_experiment` 的循环次数。

运行：

```powershell
conda activate pytorch_python11
python "Experimental control.py"
```


### Windows / Linux 动态库说明

- Windows：默认加载
  - `CPython_GeneratingSequences\Only_AffinityMatrix_efficient.dll`
  - `CPython_GeneratingSequences/alpha_AffinityMatrix_efficient.dll`
- Linux：同目录下提供 `.so` 文件。如在 Linux/WSL 运行，请将 `ctypes.CDLL(...)` 的路径改为对应的 `.so` 文件路径。
- 如果动态库加载失败，请确认工作目录为项目根目录，并检查文件名与分隔符（Windows 下 `\\`、`/` 都可被 Python 识别，但建议统一）。
- linux上要是导入失败了重新编译一下就行。


### 常见问题（FAQ）

- Q1：没有 GPU 能跑吗？
  - 需要改动：`RWR_UIL.py` 主流程中有多处 `.cuda()` 调用（如 `source_vector = source_vector.cuda()`、`mapping_model = mapping_model.cuda()` 等）。如果仅 CPU 环境，请删除或注释这些 `.cuda()` 调用，并将 `RWR_DeepWalk(..., device='cuda')` 改为 `device='cpu'`。

- Q2：如何只用纯 RWR 或只用亲和矩阵？
  - 在 `parameter_setting.py` 里设置 `solution=0.1`（纯 RWR）或 `solution=0.2`（亲和矩阵 + C 加速）。

- Q3：评估指标如何解释？
  - `evaluate.py` 中实现了贪心匹配准确率、MAP、AUC、Hit、Precision@k 等常用指标，主程序将分别在训练集/测试集上打印。

- Q4：训练/验证/测试如何划分？
  - 在 `RWR_UIL.py` 中，`groundtruth` 会按 `train_ratio` 和 `val_ratio` 划分（剩余为测试）。

- Q5：能否开启网络结构增强？
  - `RWR_UIL/RWR_get_stg.py` 中已放置三元闭包、删叶子节点、网络增强的代码，默认注释。按需取消注释，但需确保负样本、节点索引与后续流程仍一致。


### 参考运行命令速查

```powershell
# 1) 启环境
conda activate pytorch_python11

# 2) 安装依赖（示例）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy scikit-learn networkx gensim tqdm

# 3) CUDA 自检（可选）
python CudaTest.py

# 4) 单次实验（默认 douban + solution=23）
python RWR_UIL.py

# 5) 批量实验（会反复改写 parameter_setting.py 并多次运行）
python "Experimental control.py"
```


### 备注

- 随机种子默认固定为 616（NumPy、Python、PyTorch），以提高复现性。
- 默认启用 `CUDA_VISIBLE_DEVICES=0`。如需切换 GPU，修改 `RWR_UIL.py` 顶部的环境变量或按需设置。


