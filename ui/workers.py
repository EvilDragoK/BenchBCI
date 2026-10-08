"""后台任务线程：一切 >1s 的操作放这里，GUI 不卡死。"""
from __future__ import annotations

import traceback

from PyQt6.QtCore import QThread, pyqtSignal

#: 常见异常名 → 中文提示
_EXC_ZH = {
    "FileNotFoundError": "文件不存在",
    "ValueError": "数值错误",
    "TypeError": "类型错误",
    "KeyError": "键不存在",
    "IndexError": "索引越界",
    "AttributeError": "属性不存在",
    "RuntimeError": "运行错误",
    "ModuleNotFoundError": "缺少依赖模块",
    "ImportError": "导入失败",
    "ZeroDivisionError": "除零错误",
    "MemoryError": "内存不足",
    "PermissionError": "没有权限",
}


def exc_to_zh(exc: BaseException) -> str:
    """把异常转换为中文提示。"""
    name = type(exc).__name__
    return f"{_EXC_ZH.get(name, name)}：{exc}"


class FunctionThread(QThread):
    ok = pyqtSignal(object)
    fail = pyqtSignal(str)
    logged = pyqtSignal(str)

    def __init__(self, fn, *args, parent=None, **kwargs):
        super().__init__(parent)
        self._fn, self._args, self._kwargs = fn, args, kwargs
        self.stop_event = None
        import threading

        self.stop_event = threading.Event()

    def _emit_log(self, text):
        self.logged.emit(str(text))

    def stop(self):
        if self.stop_event is not None:
            self.stop_event.set()

    def run(self):
        try:
            kw = dict(self._kwargs)
            sig = self._fn.__code__.co_varnames[: self._fn.__code__.co_argcount]
            if "log" in sig and "log" not in kw:
                kw["log"] = self._emit_log
            if "stop_event" in sig and "stop_event" not in kw:
                kw["stop_event"] = self.stop_event
            res = self._fn(*self._args, **kw)
            self.ok.emit(res)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.fail.emit(exc_to_zh(e))
