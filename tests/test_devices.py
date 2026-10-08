"""训练设备：探测 / 解析（含多卡）/ 测速 / 多卡包装与权重键前缀。

CUDA 部分用打桩模拟（本机是 macOS + MPS，没有真实显卡），
保证多卡配置在 Linux/Windows 上也能正确解析。
"""
from __future__ import annotations

import os

import pytest
import torch

from core.train.devices import (adapt_state_dict, device_label, list_devices,
                                probe_device, resolve_device, system_info,
                                unwrap_state_dict, wrap_multi_gpu)


# ------------------------------------------------------------------ 探测
def test_list_devices_always_has_cpu():
    devs = list_devices()
    ids = [d.id for d in devs]
    assert "cpu" in ids
    assert len(ids) == len(set(ids))
    kinds = {d.kind for d in devs}
    assert kinds <= {"cpu", "cuda", "mps", "xpu"}
    # 有 CUDA 时每张卡都应出现
    assert sum(1 for d in devs if d.kind == "cuda") == torch.cuda.device_count()


def test_list_devices_reflects_mps():
    devs = {d.id: d for d in list_devices()}
    mps_ok = bool(getattr(torch.backends, "mps", None)
                  and torch.backends.mps.is_available())
    assert ("mps" in devs) == mps_ok


def test_system_info_has_key_fields():
    info = system_info()
    for key in ("系统", "Python", "PyTorch", "CUDA 可用", "MPS 可用", "设备数"):
        assert key in info and str(info[key])


# ------------------------------------------------------------------ 解析
def test_resolve_cpu_explicit():
    assert resolve_device("cpu") == ("cpu", [])
    assert resolve_device("CPU") == ("cpu", [])
    assert resolve_device(None)[0] in {d.id for d in list_devices()}


def test_resolve_auto_picks_accelerator():
    prim, ids = resolve_device("auto")
    kinds = {d.id: d.kind for d in list_devices()}
    if "cuda:0" in kinds:
        assert prim == "cuda:0" and ids == [0]
    elif "mps" in kinds:
        assert prim == "mps" and ids == []
    else:
        assert (prim, ids) == ("cpu", [])


def test_resolve_unavailable_device_falls_back():
    """配置里写了 cuda 但在没显卡的机器上跑 → 优雅回退，不报错。"""
    if torch.cuda.is_available():
        pytest.skip("本机有 CUDA，不适用回退分支")
    prim, ids = resolve_device("cuda")
    assert prim != "cuda" and prim in {d.id for d in list_devices()}
    assert resolve_device("cuda:0,1") == resolve_device("auto")


def test_resolve_mps_only_if_available():
    prim, _ = resolve_device("mps")
    mps_ok = bool(getattr(torch.backends, "mps", None)
                  and torch.backends.mps.is_available())
    assert (prim == "mps") == mps_ok


@pytest.fixture()
def fake_4gpus(monkeypatch):
    """假装有 4 张卡（用于验证多卡写法，不需要真显卡）。"""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    # 属性要够 nn.DataParallel 的内部检查用（multi_processor_count 等）
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i: type("P", (), {"name": f"Fake GPU {i}",
                                                 "total_memory": 24 * 1024 ** 3,
                                                 "multi_processor_count": 1,
                                                 "major": 8, "minor": 0})())
    return 4


def test_resolve_multi_gpu_specs(fake_4gpus):
    assert resolve_device("cuda") == ("cuda:0", [0])
    assert resolve_device("cuda:2") == ("cuda:2", [2])
    assert resolve_device("cuda:0,1") == ("cuda:0", [0, 1])
    assert resolve_device("cuda:0 cuda:1") == ("cuda:0", [0, 1])
    assert resolve_device("cuda:1,0") == ("cuda:0", [0, 1])       # 排序去重
    assert resolve_device("0,3") == ("cuda:0", [0, 3])
    assert resolve_device("cuda:9") == ("cuda:0", [0])            # 越界 → 回 0 号
    assert device_label("cuda:0,1").startswith("2× Fake GPU 0")


def test_wrap_multi_gpu_only_for_multiple_cuda(fake_4gpus, monkeypatch):
    from torch import nn

    model = nn.Linear(4, 2)
    # 单卡 / 非 CUDA → 原样返回
    assert wrap_multi_gpu(model, [], "cpu") is model
    assert wrap_multi_gpu(model, [0], "cuda:0") is model
    assert wrap_multi_gpu(model, [0, 1], "mps") is model
    # 多卡 → DataParallel（不打桩时 nn.DataParallel 构造不要求真显卡）
    wrapped = wrap_multi_gpu(model, [0, 1], "cuda:0")
    assert isinstance(wrapped, nn.DataParallel)
    assert wrapped.device_ids == [0, 1]
    # 卡数不足时退回单卡
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert wrap_multi_gpu(model, [0, 1], "cuda:0") is model


def test_state_dict_prefix_roundtrip(fake_4gpus):
    from torch import nn

    plain = nn.Linear(3, 2)
    wrapped = nn.DataParallel(plain, device_ids=[0, 1])
    sd_plain = plain.state_dict()
    sd_wrapped = wrapped.state_dict()
    assert all(k.startswith("module.") for k in sd_wrapped)

    # 保存时统一去掉前缀 → 与设备无关
    assert set(unwrap_state_dict(sd_wrapped)) == set(sd_plain)
    # 载入时按当前模型自适应
    assert all(k.startswith("module.") for k in adapt_state_dict(sd_plain, wrapped))
    assert not any(k.startswith("module.")
                   for k in adapt_state_dict(sd_wrapped, plain))
    # 单卡模型载入已去前缀的权重 → 原样
    assert set(adapt_state_dict(sd_plain, plain)) == set(sd_plain)


# ------------------------------------------------------------------ 测速
def test_probe_device_cpu_ok():
    res = probe_device("cpu", size=64, steps=2)
    assert res["ok"] and res["device"] == "cpu"
    assert res["ms_per_step"] is not None and res["ms_per_step"] > 0
    assert res["error"] == ""


def test_probe_device_reports_error(monkeypatch):
    """设备不可用时要给出明确错误，而不是抛异常。"""
    monkeypatch.setattr("core.train.devices.resolve_device",
                        lambda spec=None: ("cuda:9", [9]))
    res = probe_device("cuda:9", size=8, steps=1)
    assert res["ok"] is False and res["error"]


def test_probe_device_auto_ok_on_this_machine():
    res = probe_device("auto", size=64, steps=1)
    assert res["ok"], res["error"]
    assert res["name"]


# ------------------------------------------------------------------ 界面
def test_device_dialog_and_train_page(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])   # noqa: F841

    from ui.main_window import AppState
    from ui.train_page import TrainPage
    from ui.widgets.device_dialog import DeviceDialog

    dlg = DeviceDialog(current="cpu")
    assert dlg.selected_spec() == "cpu"
    dlg.chk_auto.setChecked(True)
    assert dlg.selected_spec() == "auto"
    assert dlg.tbl.rowCount() == len(list_devices())

    page = TrainPage(AppState())
    # 下拉框至少含「自动」+ 本机设备
    specs = [page.combo_device.itemData(i) for i in range(page.combo_device.count())]
    assert "auto" in specs and "cpu" in specs
    assert page.combo_device.currentData() == "auto"
    assert "实际使用" in page.lbl_device.text()

    # 选设备 → 写进配置 → 读回一致
    page._select_device("cpu")
    cfg = page.build_config()
    assert cfg["trainer"]["device"] == "cpu"
    page._populate_devices(keep="auto")
    page._apply_config(cfg)
    assert page.combo_device.currentData() == "cpu"

    # 配置里的多卡写法在下拉框里没有时也能回显
    page._populate_devices(keep="cuda:0,1")
    assert page.combo_device.currentData() == "cuda:0,1"
