# BCI Lab — EEG 预处理与深度学习训练平台

一个桌面 GUI + 命令行双入口的 BCI（脑机接口）应用，覆盖 **数据导入 → 预处理 → 深度学习训练 → 结果分析 → 统计分析** 的完整闭环。UI 与算法完全分离：`core` 层不依赖 PyQt，可被脚本/命令行独立调用，保证实验可复现。

> 📖 **完整功能清单、每个功能的实测数据、以及一轮完整正确性审计（17 项已修 bug）见 [`docs/功能说明.md`](docs/功能说明.md)**；
> 其中所有实测数字由 `.venv/bin/python docs/demo_features.py | sed "s|$HOME|~|g" > docs/demo_output.txt` 真跑产出（用 sed 把家目录缩写为 `~`，避免把本机用户名写进仓库）（120 项检查，原始输出 `docs/demo_output.txt`）。
>
> 🚀 **换机器 / 上传 GitHub / Windows 环境配置：见 [`docs/GitHub上传与Windows环境配置.md`](docs/GitHub上传与Windows环境配置.md)**
> （含：上传前检查 → `git push` → Windows 克隆 → venv → 依赖 → **只设一个环境变量 `BCI_DATA_ROOT` 即可换数据目录** → 自检 → 常见报错排查）

技术栈：PyQt6 + pyqtgraph（界面）、MNE-Python（EEG）、PyTorch（深度学习）、scikit-learn（传统基线）、YAML（配置）、SQLite（实验记录）。

---

## 1. 快速开始

### 1.1 创建虚拟环境并安装依赖（跨平台：Windows / macOS / Linux）

```bash
cd ~/Desktop/APP/BCIAPP
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> 中国大陆网络可用镜像加速：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

### 1.2 启动 GUI

```bash
python main.py
```

操作顺序：**数据管理 → 预处理工作台 → 训练控制台 → 结果分析**。

### 1.3 命令行（与 GUI 完全同构，论文实验可脚本化复现）

```bash
# 1) 预处理：读取 BNCI2014001 → 生成 Epochs FIF + 训练张量 .npy
python -m core.pipeline configs/preprocess_bnci2014001.yaml

# 2) 训练传统基线 CSP+LDA（session 划分：逐被试 T 训练 / E 测试）
python -m core.cli train configs/train_csp_lda.yaml

# 3) 训练 EEGNet（可 --epochs N / --device cpu|cuda|mps 覆盖）
python -m core.cli train configs/train_eegnet.yaml --epochs 60 --device mps

# 4) 查看实验 / 摘要
python -m core.cli list
python -m core.cli report experiments/<exp_dir>
```

---

## 2. 数据集（BNCI2014001 = BCI Competition IV 2a）

- 路径：`~/Desktop/DL/BCI/data/BNCI2014001`（可用环境变量 `BCI_DATA_ROOT` 覆盖），文件 `A01T.mat` ~ `A09E.mat`（9 名被试 × T/E 两个 session）。
- 每个 `.mat` 含 9 个 run，`X`(时间×25通道, µV) / `trial`(试次起点) / `y`(1~4) / `fs=250` / `classes` / `artifacts`。
- 22 个 EEG 电极 + 3 个 EOG；4 类：`left_hand / right_hand / feet / tongue`。

> **关键细节**：BNCI2014-001 的 `trial` 字段标记「试次起点（fixation cross）」，cue 箭头在其后 **2 s**。因此加载器把事件标注放在 `trial + 2 s`，标准 MI 分段 `cue 后 0.5~2.5 s` 即对应 `trial 后 2.5~4.5 s`（经验上判别信号最强处）。该偏移由 `core/io/bnci.py` 中的 `CUE_OFFSET_S` 控制。

预处理默认流水线（`configs/preprocess_bnci2014001.yaml`）：
`设置导联 → 仅保留 EEG → 50Hz 陷波 → 4–40Hz 带通(FIR 零相位) → 平均参考(CAR) → 分段(0.5–2.5s) → 振幅阈值剔除(150µV)`。

可用的 19 个算子（默认只启用前 7 个，其余按需勾选）：
`set_montage / pick_eeg / drop_channels / mark_bad_channels / interpolate_bads / notch_filter /
bandpass_filter / gaussian_filter / resample / normalize / re_reference（CAR·指定·双极·REST·投影）/
reject_bad_segments（连续坏段）/ clip_outliers（去极值）/ artifact_regression（回归去眼动）/
ica（含 5 判据自动识别伪迹成分）/ epoching / amplitude_reject / reject_outlier_epochs（IQR 离群）/
euclidean_alignment（逐被试 EA）`。

---

## 3. 目录结构

```
BCIAPP/
├── main.py                  # GUI 入口（--selftest 无头自检）
├── requirements.txt
├── configs/                 # 预置流水线与训练配置（YAML）
├── core/                    # 核心层（纯 Python，不依赖 PyQt）
│   ├── io/                  #   BNCI2014001 加载器 + 通用格式读取
│   ├── preprocess/          #   预处理算子（每步一个类，schema 驱动 GUI 表单）
│   ├── pipeline.py          #   流水线编排器 + 命令行入口
│   ├── datasets.py          #   张量数据集、标准化(防泄漏)、EA、增强
│   ├── models/              #   EEGNet / ShallowConvNet / DeepConvNet / FBCNet / EEGConformer / CSP 基线
│   ├── train/               #   划分、训练循环、评估、实验报告
│   ├── experiment.py        #   实验管理器（目录 + SQLite 索引）
│   └── cli.py               #   统一 CLI（pipeline/train/list/report）
├── ui/                      # 界面层（只做展示与事件，不写算法）
│   ├── main_window.py       #   主窗口 + 可折叠步骤栏 + 主题（5 页）
│   ├── data_page.py         #   数据管理页（加载/信号浏览/PSD/特征提取）
│   ├── preprocess_page.py   #   预处理工作台（19 算子流水线 + 即时预览）
│   ├── train_page.py        #   训练控制台（模型/划分/设备/自定义模型 + 实时曲线 + 论文超参抽取）
│   ├── result_page.py       #   结果分析页（混淆矩阵/多实验对比/批量删除）
│   ├── analysis_page.py     #   信号分析页（频域/ERP/时频/非线性熵/脑网络 + 协变·变量关联工作台）
│   ├── theme.py             #   主题（默认浅紫粉 + 3 套备选）
│   └── widgets/             #   15 个功能对话框（信号浏览器、ICA 浏览器、模型编辑器…）
├── docs/                    # 功能说明.md + demo_features.py（演示与自检）+ 演示输出 + 上传/Windows 配置手册
├── scripts/                 # 运维小工具（diagnose_github.py：诊断 push 失败：TCP/TLS/SSH 三段）
├── tests/                   # pytest（361 项，全部可无 GPU 跑）
├── processed/               # （运行时生成）预处理输出：X/y/sid/sess.npy + FIF
└── experiments/             # （运行时生成）实验目录 + index.db
```

---

## 4. 关键设计

- **UI 与算法分离**：`core` 不 import PyQt；同一条流水线在 GUI 与 `python -m core.pipeline` 下结果逐点一致（`tests/test_pipeline.py::test_pipeline_deterministic` 断言）。
- **防数据泄漏**：逐通道 z-score 的均值/方差**只用训练折统计量**；CSP/ICA 拟合、类别加权权重均严格限制在训练折。`fit` 与 `transform` 分离。
- **标准 BCI 评测协议**：`session` 划分 = 逐被试 T 训练 / E 测试（多被试聚合为「均值±标准差」）；支持 `kfold`（被试内）、`loso`（留一被试）、`holdout`。
- **可复现**：每个实验自动生成 `experiments/<ts>_<model>_<name>/`（`config.yaml` + `log.txt` + `checkpoints/` + `metrics.csv` + `summary.json` + `report.md`），凭配置即可原样复现；随机种子 + 可选 `torch.use_deterministic_algorithms`。
- **训练可视化**：后台线程训练，GUI 每 epoch 刷新 train/val loss 与 accuracy；断点续训（每 epoch 存 checkpoint，含 model/optimizer/scheduler/RNG 状态）。
- **设备自适应**：`cuda > mps > cpu` 自动选择，AMP 仅在 CUDA 开启。

---

## 5. 已验证结果（本机真实数据）

| 模型 | 划分 | 结果 |
|---|---|---|
| CSP + LDA（`csp_lda`） | session（逐被试 T→E，9 折） | **acc 58.18% ± 14.91%**，kappa 0.442 ± 0.199，MCC 0.456，**ITR 3.12 ± 2.31 bits/min**（单被试 37.5%~77.1%） |
| EEGNet（`--epochs 3` 冒烟） | session | acc 26.23%（**未收敛**，仅验证训练链路：MPS 设备、每 epoch 0.06 s、val 正确切出） |

> 单测 **347 项**全部通过；功能演示/自检 `docs/demo_features.py` **120 项全部通过**；
> GUI 无头自检（`QT_QPA_PLATFORM=offscreen python main.py --selftest`）通过。
> 数据集规模：BNCI2014001 9 被试 × T/E → `processed/bnci2014001` 共 **5184 epochs**。

---

## 6. 配置说明（片段）

`configs/train_eegnet.yaml` 关键项：

```yaml
model:  {name: eegnet, params: {F1: 8, F2: 16, D: 2, kernel_length: 125, dropout: 0.25}}
split:  {scheme: session, k: 5, val_frac: 0.15, seed: 42}   # session|kfold|loso|holdout
normalization: {mode: zscore_channel, euclidean_alignment: false}
trainer: {epochs: 60, batch_size: 64, lr: 0.001, optimizer: adamw, scheduler: cosine, patience: 12}
```

`configs/train_csp_lda.yaml`：`model.name: csp_lda`（也可 `csp_svm` / `fbcsp_lda`）。

---

## 7. 模型清单

- **EEGNet**（Lawhern 2018）、**ShallowConvNet / DeepConvNet**（Schirrmeister 2017）、**FBCNet**（Mane 2021 精简复现）、**EEG Conformer**（Song 2022）。
- 传统基线：**CSP+LDA / CSP+SVM / FBCSP+LDA**（sklearn 接口，与 PyTorch 路径统一）。
- **EEG 基础模型（16 个）**：LaBraM / CBraMod / EEGPT / BIOT / BENDR / Signal-JEPA / Brant / BrainBERT / EEGDINO / CodeBrain / LUNA / NeuroGPT / BrainOmni / EEGMamba / SINGLE / TFMTokenizer（经 braindecode 接入，支持 `pretrained: auto` 自动下载权重 + 冻结/分层学习率微调）。
- **自定义模型**：训练控制台「自定义模型…」→ 以内置模型定义为模板改写 → **检查（真跑前向、逐层维度追踪、反传、类别数核对）** → 保存为 `models_custom/<名字>.py`，立即出现在模型下拉框。
- 特征/归一化：CSP、逐通道 z-score、逐 epoch minmax、欧几里得对齐（EA，逐被试）。
- ⚠️ 各模型对输入时长的最低要求（采样点）：EEGNet/FBCNet/Conformer ≥ 64、ShallowConvNet ≥ 128、**DeepConvNet ≥ 500**。

---

## 8. 跨平台注意事项

- 依赖均提供 Windows/macOS/Linux 轮子；GPU 训练 Windows/Linux 用 CUDA，macOS 自动用 MPS，均退化为 CPU。
- 路径优先用配置中的绝对路径；代码内部对相对路径做了项目根目录解析，换机只需改 `configs/*.yaml` 中的 `root` / `processed_dir`。
- GUI 依赖显示服务（Linux 服务器需 X/Wayland 或 `QT_QPA_PLATFORM=offscreen` 做无头测试）。

## 9. 故障排查

- `mne/torch/PyQt6` 缺失 → 按第 1 节安装依赖。
- 训练报 `weight tensor should be defined ...` / 类别权重异常 → 已修复（按实际类别计算权重）。
- 数据不在默认路径 → 改 `configs/preprocess_bnci2014001.yaml` 的 `dataset.root` 或在 GUI 数据管理页重新填写路径。
