"""模型注册与工厂。"""
from __future__ import annotations

from .eegnet import EEGNet
from .shallowconvnet import ShallowConvNet
from .deepconvnet import DeepConvNet
from .fbcnet import FBCNet
from .conformer import EEGConformer
from .sklearn_baselines import ClassicBaseline, is_classic

TORCH_MODELS = {
    "eegnet": EEGNet,
    "shallowconvnet": ShallowConvNet,
    "deepconvnet": DeepConvNet,
    "fbcnet": FBCNet,
    "conformer": EEGConformer,
}
CLASSIC_MODELS = ("csp_lda", "csp_svm", "fbcsp_lda")
MODEL_NAMES = list(TORCH_MODELS) + list(CLASSIC_MODELS)


def custom_model_names() -> list[str]:
    """已保存的自定义模型名（models_custom/*.py，目录不存在则为空）。"""
    from .custom import custom_names

    return custom_names()


def is_custom(name: str) -> bool:
    from .custom import is_custom as _is_custom

    return _is_custom(name)


def all_model_names() -> list[str]:
    """下拉框用：内置 + 经典基线 + 自定义。"""
    return list(MODEL_NAMES) + custom_model_names()


def build_model(name: str, n_classes: int, n_ch: int, n_times: int, params: dict | None = None):
    """按名称构建模型；经典基线返回 ClassicBaseline，自定义模型走动态加载。"""
    params = dict(params or {})
    if is_classic(name):
        return ClassicBaseline(kind=name,
                               n_components=int(params.get("n_components", 6)),
                               C=float(params.get("C", 1.0)))
    if is_custom(name):
        # 自定义模型：直接从 models_custom/<name>.py 加载并实例化
        from .custom import instantiate, load_class

        model, _how = instantiate(load_class(name), n_classes, n_ch, n_times, params)
        return model
    if name not in TORCH_MODELS:
        raise KeyError(f"未知模型：{name}（可用：{all_model_names()}）")
    return TORCH_MODELS[name](n_classes=n_classes, n_ch=n_ch, n_times=n_times, **params)
