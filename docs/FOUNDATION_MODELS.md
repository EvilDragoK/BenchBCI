# EEG 基础模型（Foundation Model）接入指南

> 参考 [EEG-FM-Benchmark](https://github.com/Dingkun0817/EEG-FM-Benchmark) 的基础模型清单，
> 权重优先走 **braindecode 在 HuggingFace 上维护的 curated checkpoints**（safetensors + config）。

## 一、支持的基础模型

在训练控制台「模型」下拉框的最后一段（分隔线之后）即可选择；也可命令行 `model.name: <名字>`。

| 名字 | 模型 | 参数量* | 权重仓库（自动下载） | 备注 |
|---|---|---|---|---|
| `cbramod` | CBraMod | 5.0 M | `braindecode/cbramod-pretrained` | ✅ 实测可用，社区最常用之一 |
| `eegpt` | EEGPT | 25.3 M | `braindecode/eegpt-pretrained` | ✅ 实测可用（权重 202 MB） |
| `biot` | BIOT | 3.2 M | `braindecode/biot-pretrained-six-datasets-18chs` | ✅ 实测可用（另有 16ch / shhs 版本） |
| `bendr` | BENDR | 157.1 M | `braindecode/braindecode-bendr` | ✅ 实测可用（另有插值版） |
| `labram` | LaBraM | 5.8 M（插值版） | `braindecode/labram-pretrained` | ⚠️ 原生需 128 通道 → **自动改用 `InterpolatedLaBraM`** |
| `signaljepa` | Signal-JEPA | 3.5 M | `braindecode/SignalJEPA-pretrained` | ⚠️ 主干只输出特征 → **自动补分类头** |
| `codebrain` | CodeBrain | 29.4 M | `braindecode/codebrain-pretrained` | ✅ 实测可用 |
| `eegdino` | EEG-DINO | — | `braindecode/eegdino-small-pretrained` | ⚠️ 官方权重为 **19 通道**，22 通道需换数据/权重 |
| `luna` | LUNA | — | `braindecode/LUNA`（架构） | 权重按官方仓库另取 |
| `brant` / `brainbert` | Brant / BrainBERT | — | `braindecode/brant-pretrained` 等 | 权重仓库存在，但当前 braindecode 版本未提供类，需自定义适配 |
| `neurogpt` `brainomni` `eegmamba` `single` `tfmtokenizer` | — | — | 官方仓库 | 未收录在 braindecode，需自行接入 |

\* 参数量为「4 类 / 22 通道 / 800 采样点」下的实测值（`build_foundation_model` 自动匹配各模型的参数命名）。

## 二、预训练权重：地址与下载

`model.pretrained` 支持四种写法：

| 写法 | 含义 |
|---|---|
| `auto` | 按注册表自动从 HuggingFace 下载（**推荐**） |
| `hf:<repo>` | 下载该仓库全部权重（需 `huggingface_hub`） |
| `hf:<repo>/<file>` | 直链下载单个文件（**无需额外依赖**） |
| `url:https://...` | 任意直链 |
| `<本地路径>` | 直接使用本地文件/目录 |

缓存目录：`~/.cache/bci_lab/pretrained/`（可用环境变量 `BCI_LAB_CACHE` 覆盖）。

```bash
# CLI 方式：在配置里写 pretrained: auto 即可，训练时自动下载
# GUI 方式：模型下拉选好后点「下载/校验」，成功后自动回填本地路径
```

实测（CBraMod）：
```
下载预训练权重：braindecode/cbramod-pretrained → model.safetensors
  下载中 8.4/19.7 MB → 16.8/19.7 MB
  已保存 → ~/.cache/bci_lab/pretrained/braindecode__cbramod-pretrained/model.safetensors（19.7 MB）
  …训练时：已加载权重 model.safetensors（19.7 MB）：命中 211/211 项，未命中 2 项
```

> 加载是**宽容模式**（`strict=False`）：命中率低时会自动尝试 `backbone.` / `main_model.`
> 前缀对齐；未命中的通常是新补的分类头，属正常。

## 三、冻结策略与分层学习率

### 3.1 冻结哪些参数

```yaml
model:
  freeze: encoder        # none | encoder | all_but_head | custom
  trainable: []          # 模块名白名单（按参数名分段匹配），优先级高于 freeze
```

| 策略 | 含义 |
|---|---|
| `none` | 全部参数训练（从零训练 / 全参数微调） |
| `encoder` | 冻结编码器/主干，只训练分类头（**线性探针**） |
| `all_but_head` | 同 `encoder`（对齐 EEG-FM-Benchmark 的叫法） |
| `custom` | 只训练 `trainable` 里列出的模块，其余全部冻结 |

`trainable` 填法：`classifier, proj_out` → 命中 `classifier.weight`、`main_model.proj_out.weight`。

实测（CBraMod，`freeze: encoder`）：
```
微调策略 encoder：可训练 110,604/4,994,404 参数（2.2%）
  可训练模块示例：final_layer.1, proj_out.0
```

### 3.2 冻结部分与训练部分的学习率分开设置

```yaml
trainer:
  lr: 1e-3          # 分类头（新初始化层）—— 大学习率
  lr_encoder: 1e-5  # 编码器（预训练主干）—— 小学习率；0/不填 = 与 lr 相同
  layer_decay: 0.9  # <1 时按层数指数衰减：底层小、高层大（LaBraM/MAE 常见做法）
```

> 与参考实现的差异：EEG-FM-Benchmark 的 `optimizer.py` 只把 `lr_scale` 写进参数组
> **并不生效**（需要外部再乘一次）；本项目**直接写进 `group["lr"]`**，并借助
> `CosineAnnealingLR` 的 per-group `base_lrs` 让比例在整个训练过程中保持。

实测（CBraMod，`freeze: none`、`lr=1e-3`、`lr_encoder=1e-5`、`layer_decay=0.9`）：
```
head:lr=1.00e-03,162175参数, head:lr=1.00e-03,1029参数,
encoder(layer0 ×0.31):lr=3.14e-06,400000参数,
encoder(layer1 ×0.35):lr=3.49e-06,
…
encoder(layer11×1.00):lr=1.00e-05,400000参数
```
即底层 `1e-5×0.9¹¹ = 3.14e-6`、顶层 `1e-5`，**逐层指数增长**；分类头独立用 `1e-3`。

## 四、完整配置示例

见 `configs/train_cbramod_finetune.yaml`：

```yaml
experiment: {name: cbramod_finetune, note: "CBraMod 预训练 + 冻结编码器 + 分层学习率"}
data: {processed_dir: processed/my_200hz_data}
model:
  name: cbramod
  pretrained: auto          # 自动下载 braindecode/cbramod-pretrained
  freeze: encoder           # 冻结编码器，只训练分类头
trainer:
  lr: 1.0e-3                # 分类头
  lr_encoder: 1.0e-5        # 编码器（冻结时无 encoder 组，自动忽略）
  layer_decay: 1.0
  epochs: 30
  batch_size: 64
```

命令行：`python -m core.cli train configs/train_cbramod_finetune.yaml`

## 五、数据要求（重要）

基础模型多在特定采样率/patch 上预训练，**输入长度与采样率需匹配**：

| 模型 | 采样率 | 备注 |
|---|---|---|
| CBraMod | 200 Hz | patch = 200 点（1 s），`n_times` 建议为 200 的整数倍 |
| LaBraM | 200 Hz | patch = 200 点，原生 128 通道（本项目自动用插值版支持任意通道） |
| EEGPT / BIOT / BENDR | 256 Hz 左右 | 见各自官方说明 |

本项目当前 BNCI2014001 数据为 **250 Hz / 501 点**——用基础模型前请先在
「预处理工作台」加**重采样到 200 Hz** 并调整分段长度（例如 0.5–2.5 s → 400 点），
或另建一份符合要求的 `processed` 目录（`data.processed_dir` 指向它即可）。

## 六、GUI 用法（训练控制台）

| 控件 | 作用 |
|---|---|
| 模型下拉（分隔线以下的项） | 选择基础模型，选中后会把权重框自动填成 `auto` 并打印权重仓库 |
| 预训练权重 + 「下载/校验」 | 下载并缓存权重，成功后回填本地路径 |
| 冻结策略 / 可训练 | 选择 `none`/`encoder`/`all_but_head`/`custom`，并填可训练模块 |
| 编码器分层学习率 | 编码器 lr 与逐层衰减系数 |
| 查看模型定义 | 直接查看 braindecode 里该基础模型的网络源码 |

## 七、依赖

```bash
pip install braindecode            # 基础模型实现（含 Interpolated* 变体）
pip install huggingface_hub        # 下载整仓权重时需要（单文件直链不需要）
pip install safetensors einops     # 权重格式与模型实现依赖
```
