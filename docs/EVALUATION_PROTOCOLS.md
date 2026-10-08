# 划分协议（评测方案）详细对照表

> 数据：BNCI2014001（BCI Competition IV 2a），9 名被试，5184 个 trial（22 通道 × 501 采样点）
> 模型：CSP + LDA（`n_components=6`），未做超参调优
> 全部数字为**实测值**，命令见各节 `命令行`

## 一、总览（12 种协议）

> 注：`within` / `few_shot` 的折数 = **被试数 × `n_repeats`**；`n_repeats` 默认 **1**，
> 即**每名被试一折**（共 9 折），与 EEG-FM-Benchmark 等主流基准一致。

| # | 协议名 | 中文名 | 切分对象 | 折数 | 第 1 折构成 | 回答什么问题 | 实测 acc |
|---|---|---|---|---|---|---|---|
| 1 | `within` | 被试内划分（每被试**一折**） | trial（被试内） | 9 | train 391 / val 69 / test 116 | 被试内随机划分的真实水平 | **63.89%** |
| 2 | `within` + `train_percentage: 0.3` | 被试内少样本（对齐 EEG-FM-Benchmark） | trial（被试内） | 9 | train 146 / val 26 / test 404 | 少量校准数据够不够用 | **58.14%** |
| 3 | `kfold` | 逐被试 k 折 | trial（被试内） | 45 | train 391 / val 69 / test 116 | 被试内性能上限 | **64.80%** |
| 4 | `block` | 按记录块划分 | 记录块（run） | 45 | 9 人 × 5 组块 | 排除"相邻试次泄漏"后的真实水平 | **64.06%** |
| 5 | `nested_cv` | 嵌套交叉验证 | 外层折 + 内层选参 | 45 | train 391 / test 116 | 指标无偏（超参不看测试集） | **64.20%** |
| 6 | `session` | 跨会话（标准基准） | session（同人） | 9 | train 288 / val 44 / test 288 | 换个时间还认不认得 | **58.18%** |
| 7 | `session_kfold` | 同人跨会话双向 | session（同人） | 18 | train 288 / test 288 | T→E 与 E→T 都测一遍 | **56.39%** |
| 8 | `few_shot` | 每类 n 个样本校准 | trial（被试内） | 9 | train 34 / test 536 | 现场校准要多少数据 | **56.18%**（10 shot） |
| 9 | `cross_dataset` | 跨数据集迁移 | 数据集（域） | 1 | train 2592 / test 2592 | 换数据集还能不能用 | **45.02%** |
| 10 | `holdout` | 全体随机划分 | 全体 trial | 1 | train 3524 / test 1037 | 快速调试 | 47.44% |
| 11 | `loso` | 留一被试 | 被试 | 9 | train 4608 / test 576 | 换个人还能不能用 | **40.10%** |
| 12 | `subject_kfold` | 跨被试分组 k 折 | 被试 | 5 | train 3427 / test 1152 | LOSO 的 k 折提速版 | **39.31%** |
| 13 | `custom` | 自定义折 | 手工指定受试者 | 自定义 | 见配置 | 复现论文/特定对照实验 | **39.32%**（示例 5 折） |

**性能梯度**：被试内（64–65%） ＞ 跨会话（56–58%） ＞ 跨域（45–47%） ＞ 跨被试（39–40%）。
同一份数据、同一个模型，**协议不同分数相差 25 个百分点** —— 报告结果时必须写明协议。

## 一·补、与 EEG-FM-Benchmark 的协议对照

参考仓库 [`Dingkun0817/EEG-FM-Benchmark`](https://github.com/Dingkun0817/EEG-FM-Benchmark)
（`utils/dataset_split.py`）只提供 **3 类划分**，本项目已完全对齐并做了扩展：

| 参考实现 | 其做法（源码） | 本项目对应 | 说明 |
|---|---|---|---|
| `split_dataset_cross` + `split_method='LOSO'` | **每个被试一折**，train=其余被试，test=该被试 | `loso` | 折数 = 被试数 ✓ |
| `split_dataset_cross` + `split_method='K-fold'`（如 `5-fold`） | 在**被试列表**上 KFold，每折留出一组被试 | `subject_kfold` | 折数 = k ✓（补了 `random_state`，可复现） |
| `split_dataset_fewshot`（`--task_mode Fewshot`） | **每个被试一折**（`fold_idx=subject_idx`）；按 `train_percentage`（默认 **0.3**）**每被试每类**抽训练样本 | `within`（`train_percentage`）与 `few_shot`（`n_shot`） | 折数 = 被试数 ✓ |
| —（无） | — | `block` `nested_cv` `cross_dataset` `custom` `kfold` `session` `session_kfold` `holdout` | 本项目的扩展协议 |

两处实现差异（本项目的做法更严格）：

| 项 | 参考实现 | 本项目 |
|---|---|---|
| 验证集 | 无独立 val mask，在训练内部临时切（返回 `best_val_acc`） | `val_frac` 从训练集内划出，单列为 `val`，绝不碰测试集 |
| few-shot 抽样 | 类别内按原顺序取前 N（**无 shuffle**，可能带采集顺序偏置） | 按类别分层**随机**抽取（`n_repeats>1` 可多次重采样取平均） |
| K 折可复现性 | `KFold(shuffle=True)` 未设 `random_state`（每次不同） | 固定 `seed`，完全可复现 |

> 简而言之：**参考基准只有"跨被试 LOSO / 跨被试 K 折 / 被试内比例划分"三种**；
> 本项目把它的 `Fewshot` 拆成了参数化更明确的 `within`（比例 0–1）与 `few_shot`（每类固定个数）。

## 二、按 MOABB 生态分类

MOABB（Mother of All BCI Benchmarks）的官方三分法与本项目的对应关系：

| MOABB 类名 | paradigm | 本项目协议 | 说明 |
|---|---|---|---|
| `WithinSessionEvaluation` | `within-session` | `within` `kfold` `block` `few_shot` | 同被试、同 session 内切 trial |
| `CrossSessionEvaluation` | `cross-session` | `session` `session_kfold` | 同被试、换 session |
| `CrossSubjectEvaluation` | `cross-subject` | `loso` `subject_kfold` | 换被试（LOSO） |
| —（生态扩展） | — | `nested_cv` `cross_dataset` `custom` `holdout` | 无偏评估 / 域迁移 / 手工折 |

## 三、参数说明

| 参数 | 适用协议 | 含义 | 默认 |
|---|---|---|---|
| `k` | `kfold` `within` `subject_kfold` `block` `nested_cv`(外层) `holdout` | 折数；`holdout` 中表示测试占比 `1/k`；`within` 未给 `train_percentage` 时表示测试占比 `1/k` | 5 |
| `n_repeats` | `within` `few_shot` | 重复随机采样次数。**1 = 每名被试一折（共 N 折）**；>1 时折数 = 被试数 × `n_repeats` | **1** |
| `train_percentage` | `within` `few_shot` | 按「每被试每类」抽取训练样本的比例（**对齐 EEG-FM-Benchmark 的 `--train_percentage`，其默认 0.3**）；`None`/0 表示不用比例 | `None` |
| `n_shot` | `few_shot` | 每类用于训练的 trial 数（未给 `train_percentage` 时生效） | 5 |
| `val_frac` | 全部 | 从**训练集内部**划出验证集的比例（用于早停/选参） | 0.15 |
| `test_session` | `loso` `few_shot` | 测试数据取哪个 session（`all`/`T`/`E`） | `all` |
| `inner_k` | `nested_cv` | 内层折数 | 3 |
| `param_grid` | `nested_cv` | 内层挑选的超参候选，如 `{n_components: [4,6,8]}` | 空 |
| `folds` | `custom` | 每折的 `{name, train[], val[], test[]}`（受试者号） | 空 |
| `data.test_processed_dir` | `cross_dataset` | 目标域预处理目录 | 空 |

## 四、配置示例

### 0. 被试内划分（`within`）——对齐 EEG-FM-Benchmark 的 Fewshot

```yaml
split:
  scheme: within
  train_percentage: 0.3   # 每被试每类抽 30% 作训练（该基准 --task_mode Fewshot 默认值）
  n_repeats: 1            # 1 = 每被试一折（共 9 折）
  val_frac: 0.15          # 验证集从训练集内部划出
```

- 命令行：`python -m core.cli train configs/train_within_subject.yaml`
- 不写 `train_percentage` 时按 `1/k` 分层随机切（`k` 默认 5 → 每被试测试 20%）
- 该基准的 `--task_mode Cross` 对应本项目的 `loso` / `subject_kfold`

### 1. 自定义 5 折（`custom`）——每折自选训练/验证/测试受试者

```yaml
split:
  scheme: custom
  val_frac: 0.15          # 某折未填 val 时，自动从该折训练集划出
  folds:
    - {name: fold1, train: [2,3,4,5,7,8,9], val: [],      test: [1,6]}
    - {name: fold2, train: [1,3,4,5,6,8,9], val: [],      test: [2,7]}
    - {name: fold3, train: [1,2,4,5,6,7,9], val: [],      test: [3,8]}
    - {name: fold4, train: [1,2,3,5,6,7,8], val: [],      test: [4,9]}
    - {name: fold5, train: [1,2,3,6,7,8,9], val: [4],     test: [5]}   # 显式指定验证受试者
```

- GUI：训练控制台 → 划分方案选 `custom` → 点「**编辑折配置…**」→ 表格编辑（支持 `1-5` 范围写法）→「自动生成 N 折」可一键铺开再手改
- 校验：未知受试者、train/val/test 角色重叠都会**直接报错**，不会静默产生空折
- 命令行：`python -m core.cli train configs/train_custom_5fold.yaml`

### 2. 嵌套交叉验证（`nested_cv`）

```yaml
split:
  scheme: nested_cv
  k: 5                 # 外层折
  inner_k: 3           # 内层折
  param_grid: {n_components: [2, 4, 6, 8]}     # 内层挑选，不外泄到测试折
```

- 命令行：`python -m core.cli train configs/train_nested_cv.yaml`
- 日志会打印每个候选的内层平均准确率与最终选中组合

### 3. 跨数据集迁移（`cross_dataset`）

```yaml
data:
  processed_dir: processed/bnci2014001_T        # 源域（训练）
  test_processed_dir: processed/bnci2014001_E   # 目标域（测试）
split:
  scheme: cross_dataset
```

- 要求：两域通道数、epoch 时长一致；类别集合相同（标签会**自动对齐**，见第六节）
- 真实跨数据集（2a→2b、PhysioNet→2a）只需把 `test_processed_dir` 换成另一个数据集的预处理输出

### 4. 按记录块划分（`block`）

```yaml
split: {scheme: block, k: 5}     # 每名被试的多个 run 分成 5 组，轮流留出一组做测试
```

- 依赖预处理输出的 `block.npy`（新版流水线自动生成；缺失时给出明确提示）

## 五、命令行速查

```bash
python -m core.cli train configs/train_csp_lda.yaml --scheme within
python -m core.cli train configs/train_csp_lda.yaml --scheme kfold
python -m core.cli train configs/train_csp_lda.yaml --scheme block
python -m core.cli train configs/train_csp_lda.yaml --scheme session_kfold
python -m core.cli train configs/train_csp_lda.yaml --scheme loso
python -m core.cli train configs/train_csp_lda.yaml --scheme subject_kfold
python -m core.cli train configs/train_csp_lda.yaml --scheme holdout
python -m core.cli train configs/train_custom_5fold.yaml       # custom
python -m core.cli train configs/train_nested_cv.yaml          # nested_cv
python -m core.cli train configs/train_cross_dataset.yaml      # cross_dataset
python -m core.cli train configs/train_within_subject.yaml     # within + 每类 30%（对齐 EEG-FM-Benchmark）
```

## 六、实现要点与常见坑

| 要点 | 说明 |
|---|---|
| **标签重编码** | 预处理输出的类别码是 1..4，模型输出索引是 0..3。训练入口统一重编码为 `0..K-1`；否则 CPU 报 `Target 4 is out of bounds`，MPS **静默越界**导致混淆矩阵最后一列恒为 0、指标失真 |
| **跨域标签对齐** | `cross_dataset` 的**目标域**也必须做同样的重编码，否则测试标签停留在 1..4 与预测 0..3 系统性错位（症状：某真实类别整行为 0、准确率低于随机） |
| **验证集来源** | 所有协议的验证集都从**训练集内部**划出（`val_frac`），绝不触碰测试集；样本太少时自动退化为"不设验证集" |
| **逐被试切分** | `kfold` / `within` 都是**逐被试内部**切分再跨被试聚合；把多被试混在一起随机切会让同一被试的相邻 trial 同时进训练与测试（数据泄漏） |
| **block 编号全局唯一** | 每个 (被试, session) 的记录段编号带全局偏移，避免不同 session 的同号段被当成同一块 |
| **折数公式** | `within`/`few_shot`：被试数 × `n_repeats`（**`n_repeats` 默认 1 → 被试数，即每被试一折**）；`kfold`/`block`/`nested_cv`：被试数 × `k`；`session`/`loso`：被试数；`session_kfold`：被试数 × 2；`subject_kfold`/`holdout`/`cross_dataset`：`k` / 1 / 1 |
| **按被试聚合** | `per_subject` 仅在"一折对应单一被试"时生成（`loso`/`session`/`within`/`kfold`…）；`subject_kfold` 因一折含多名被试而不出柱状图 |

## 七、怎么选（决策清单）

| 目标 | 推荐协议 |
|---|---|
| 调代码 / 跑通流程 | `holdout` |
| 验证算法上限 | `kfold` 或 `within`（两者数值接近，`within` 支持重复采样） |
| 排除相邻试次泄漏 | `block` |
| 需要无偏指标（论文级） | `nested_cv` |
| 评估"隔天再用" | `session` / `session_kfold` |
| 评估"新用户免校准" | `loso`（被试少）/ `subject_kfold`（被试多） |
| 决定现场校准数据量 | `few_shot`（画学习曲线） |
| 换数据集部署 | `cross_dataset` |
| 复现论文的特定分组 | `custom` |
