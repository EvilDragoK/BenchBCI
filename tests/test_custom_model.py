"""自定义模型：模板生成 → 检查（输入/输出/逐层维度）→ 保存 → 构建 的回归测试。

其中 ``test_pyramid_variant_*`` 复现了典型用法：**以内置 EEGNet 为模板，
在 block1 与 block2 之间加入金字塔（多尺度）池化特征**，检查维度通过后再保存。
"""
from __future__ import annotations

import pytest
import torch

from core.models import build_model, is_custom
from core.models.custom import (BLANK_SOURCES, blank_template, check_code,
                                delete_custom, list_custom, save_custom,
                                sanitize_identifier, template_source)

#: 小尺寸加速：3 类 / 6 通道 / 256 采样点
DIMS = {"n_classes": 3, "n_ch": 6, "n_times": 256, "batch": 2}
PARAMS = {"kernel_length": 32, "F1": 4, "F2": 8}
NAME = "unittest_eegnet_pyramid"

PYRAMID_CLASS = '''
class PyramidPool(nn.Module):
    """金字塔（多尺度）池化：多尺度时间池化后展平拼接。"""

    def __init__(self, scales=(1, 2, 4)):
        super().__init__()
        self.scales = tuple(int(s) for s in scales)

    def forward(self, x):
        levels = [nn.functional.adaptive_avg_pool2d(x, (1, s)) for s in self.scales]
        return torch.cat([lv.flatten(1) for lv in levels], dim=1)

'''


def _pyramid_variant() -> str:
    """以内置 EEGNet 模板改写：block1 输出接金字塔特征，与 block2 特征拼接。"""
    code = template_source("eegnet", NAME, "在 block1/block2 之间加入金字塔特征")
    cls_line = f"class {NAME}(nn.Module):"
    assert cls_line in code, "模板应把类名改写为新模型名"

    code = code.replace(cls_line, PYRAMID_CLASS + cls_line, 1)
    code = code.replace(
        "        self.block2 = nn.Sequential(",
        "        # 金字塔特征：对 block1 的输出做多尺度池化\n"
        "        self.pyramid = PyramidPool((1, 2, 4))\n"
        "        self.block2 = nn.Sequential(", 1)
    code = code.replace(
        "        feat = infer_flat_dim(nn.Sequential(self.block1, self.block2), n_ch, n_times,\n"
        "                              tag=\"EEGNet\")\n"
        "        self.classifier = nn.Linear(feat, n_classes)",
        "        with torch.no_grad():\n"
        "            f1 = self.block1(torch.zeros(1, 1, n_ch, n_times))\n"
        "            feat = (self.pyramid(f1).shape[1]\n"
        "                    + self.block2(f1).flatten(1).shape[1])\n"
        "        self.classifier = nn.Linear(feat, n_classes)", 1)
    code = code.replace(
        "        return self.classifier(self.block2(self.block1(x)).flatten(1))",
        "        f1 = self.block1(x)\n"
        "        pyramid = self.pyramid(f1)          # (B, C×Σscales) 金字塔特征\n"
        "        deep = self.block2(f1).flatten(1)   # (B, F2) block2 深度特征\n"
        "        return self.classifier(torch.cat([pyramid, deep], dim=1))", 1)
    assert "self.pyramid" in code and "torch.cat" in code
    return code


# ------------------------------------------------------------------ 模板 / 检查
def test_template_from_eegnet_passes_check():
    code = template_source("eegnet", "unittest_tpl_only", "模板自检")
    res = check_code(code, params=PARAMS, **DIMS)
    assert res["ok"], res["errors"]
    assert res["output_shape"] == [DIMS["batch"], DIMS["n_classes"]]
    assert res["n_params"] > 0 and res["steps"], "应记录逐层形状与参数量"


def test_pyramid_variant_checks_saves_and_builds():
    """用户场景：EEGNet + 金字塔特征 → 检查通过 → 保存为新模型 → 可构建训练。"""
    code = _pyramid_variant()
    res = check_code(code, params=PARAMS, **DIMS)
    assert res["ok"], res["errors"]
    assert res["target_class"] == NAME
    assert [s["name"] for s in res["steps"]].count("pyramid") == 1
    assert res["output_shape"] == [DIMS["batch"], DIMS["n_classes"]]
    # 金字塔层输出的通道数 = F1*D × Σscales
    step = next(s for s in res["steps"] if s["name"] == "pyramid")
    assert step["out"][1] == PARAMS["F1"] * 2 * (1 + 2 + 4)

    try:
        path = save_custom(NAME, code, base="eegnet", note="金字塔特征", shape=DIMS)
        assert path.is_file() and is_custom(NAME)
        assert list_custom()[NAME]["base"] == "eegnet"

        model = build_model(NAME, DIMS["n_classes"], DIMS["n_ch"], DIMS["n_times"],
                            PARAMS)
        out = model(torch.randn(DIMS["batch"], 1, DIMS["n_ch"], DIMS["n_times"]))
        assert tuple(out.shape) == (DIMS["batch"], DIMS["n_classes"])
        out.sum().backward()
        assert any(p.grad is not None for p in model.parameters()), "应可反传"
    finally:
        delete_custom(NAME)
    assert not is_custom(NAME)


# ------------------------------------------------------------------ 错误报告
def test_check_reports_wrong_output_dim():
    code = template_source("eegnet", "unittest_bad_dim")
    code = code.replace("self.classifier = nn.Linear(feat, n_classes)",
                        "self.classifier = nn.Linear(feat, n_classes + 1)", 1)
    res = check_code(code, params=PARAMS, **DIMS)
    assert not res["ok"]
    assert any("输出维度" in e for e in res["errors"])


def test_check_reports_syntax_error():
    res = check_code("class A(nn.Module):\n    def forward(self, x)  return x\n", **DIMS)
    assert not res["ok"]
    assert any("语法" in e for e in res["errors"])


def test_check_reports_missing_class():
    res = check_code("x = 1 + 1\n", **DIMS)
    assert not res["ok"]
    assert any("nn.Module 子类" in e for e in res["errors"])


# ------------------------------------------------------------------ 工厂约定
def test_model_marker_and_dim_aliases():
    """__model__ 指定类名；构造函数用 num_classes / n_chans / seq_len 别名也能建。"""
    code = '''
import torch
import torch.nn as nn


class Backbone(nn.Module):
    def forward(self, x):
        return x


class Net(nn.Module):
    def __init__(self, num_classes: int = 4, n_chans: int = 22, seq_len: int = 501):
        super().__init__()
        self.head = nn.Linear(seq_len, num_classes)

    def forward(self, x):
        if x.ndim == 4:
            x = x.mean(dim=(1, 2))
        return self.head(x)


__model__ = "Net"
'''
    res = check_code(code, **DIMS)
    assert res["target_class"] == "Net"
    assert res["ok"], res["errors"]


# ------------------------------------------------------------------ 从零设计
def test_sanitize_identifier():
    assert sanitize_identifier("MyNet") == "MyNet"
    assert sanitize_identifier("my net v2") == "my_net_v2"
    assert sanitize_identifier("2net") == "M2net"
    assert sanitize_identifier("   ") == "MyEEGNet"
    assert sanitize_identifier("Net-1.2") == "Net_1_2"


@pytest.mark.parametrize("style", [k for k, _ in BLANK_SOURCES])
def test_blank_template_passes_check(style):
    """从零设计的骨架必须开箱即用地通过检查（输入→输出维度正确、可训练）。"""
    code = blank_template("MyBlankNet", n_classes=DIMS["n_classes"],
                          n_ch=DIMS["n_ch"], n_times=DIMS["n_times"], style=style)
    assert 'class MyBlankNet(nn.Module):' in code
    assert '__model__ = "MyBlankNet"' in code
    res = check_code(code, **DIMS)
    assert res["ok"], res["errors"]
    assert res["target_class"] == "MyBlankNet"
    assert res["output_shape"] == [DIMS["batch"], DIMS["n_classes"]]
    assert res["n_params"] > 0 and res["steps"]


@pytest.mark.parametrize("n_times,n_ch,n_classes", [(501, 22, 4), (256, 8, 3),
                                                    (128, 4, 2)])
def test_blank_template_various_dims(n_times, n_ch, n_classes):
    """不同数据集维度都要能通过（骨架里的分类头是自动探测的）。"""
    code = blank_template("NetDims", n_classes=n_classes, n_ch=n_ch, n_times=n_times)
    res = check_code(code, n_classes=n_classes, n_ch=n_ch, n_times=n_times, batch=3)
    assert res["ok"], res["errors"]
    assert res["output_shape"] == [3, n_classes]


def test_blank_template_name_goes_into_class_and_file():
    """输入英文模型名 → 生成同名类；保存后可被 build_model 加载。"""
    name = "unittest_blank_scratch"
    code = blank_template(name, **{k: DIMS[k] for k in ("n_classes", "n_ch", "n_times")})
    assert f'class {name}(nn.Module):' in code
    assert f'__model__ = "{name}"' in code
    # 说明会写进文件头
    code2 = blank_template(name, note="三层卷积 + 金字塔池化",
                           **{k: DIMS[k] for k in ("n_classes", "n_ch", "n_times")})
    assert "三层卷积 + 金字塔池化" in code2
    try:
        save_custom(name, code, base="", note="从零设计", shape=DIMS)
        assert is_custom(name)
        model = build_model(name, DIMS["n_classes"], DIMS["n_ch"], DIMS["n_times"], {})
        out = model(torch.randn(DIMS["batch"], 1, DIMS["n_ch"], DIMS["n_times"]))
        assert tuple(out.shape) == (DIMS["batch"], DIMS["n_classes"])
        out.sum().backward()
    finally:
        delete_custom(name)
    assert not is_custom(name)


def test_blank_template_reports_shape_problems():
    """骨架改错时检查要明确报错：输出维度不符 / 层间接不上（前向失败）。"""
    base = blank_template("BadNet", **{k: DIMS[k] for k in ("n_classes", "n_ch",
                                                            "n_times")})
    # ① 分类头多一类 → 输出维度不对
    off = base.replace("self.classifier = nn.Linear(feat, n_classes)",
                       "self.classifier = nn.Linear(feat, n_classes + 1)")
    assert off != base
    res = check_code(off, **DIMS)
    assert not res["ok"]
    assert any("输出维度" in e for e in res["errors"])

    # ② 忘了 flatten（层间接不上）→ 前向传播失败，并带出原始报错
    broken = base.replace("        return self.classifier(h.flatten(1))",
                          "        return self.classifier(h.mean(dim=-1))")
    assert broken != base
    res2 = check_code(broken, **DIMS)
    assert not res2["ok"]
    assert any("前向传播失败" in e for e in res2["errors"])
