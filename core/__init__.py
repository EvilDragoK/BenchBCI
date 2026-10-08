"""BCI Lab 核心层：数据 IO、预处理流水线、模型、训练与实验管理。

core 层不依赖 PyQt，可独立命令行调用，例如::

    python -m core.pipeline configs/preprocess_bnci2014001.yaml
    python -m core.cli train configs/train_eegnet.yaml
"""

__version__ = "0.1.0"

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
