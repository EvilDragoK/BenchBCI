"""实验管理器：目录结构 + SQLite 索引，保证任何结果可凭 config.yaml 复现。"""
from __future__ import annotations

import re
import shutil
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

import yaml

from . import PROJECT_ROOT

_LOCK = threading.Lock()


def dir_size(path: str | Path) -> int:
    """目录占用字节数（不存在返回 0）。"""
    p = Path(path)
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    total = 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:  # pragma: no cover - 权限/文件消失
            pass
    return total


class Experiment:
    def __init__(self, root: Path, name: str):
        self.root = root
        self.dir = root / name
        self.ckpt_dir = self.dir / "checkpoints"
        self.figures_dir = self.dir / "figures"
        for d in (self.dir, self.ckpt_dir, self.figures_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.log_path = self.dir / "log.txt"

    # ------------------------------------------------------------------
    def write_config(self, config: dict):
        with open(self.dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    def log(self, text: str):
        with _LOCK:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(text.rstrip("\n") + "\n")

    def save_metrics(self, rows: list[dict]):
        import csv

        if not rows:
            return
        keys = sorted({k for r in rows for k in r})
        with open(self.dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    def save_summary(self, summary: dict):
        import json

        with open(self.dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)


class ExperimentManager:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else PROJECT_ROOT / "experiments"
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "index.db"
        self._init_db()

    def _conn(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS experiments(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT, name TEXT, dataset TEXT, model TEXT, scheme TEXT,
                dir TEXT, n_trials INTEGER,
                acc_mean REAL, acc_std REAL, kappa_mean REAL, note TEXT)""")

    # ------------------------------------------------------------------
    def create(self, name: str, config: dict) -> Experiment:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = re.sub(r"[^\w\-.]+", "_", name).strip("_") or "exp"
        exp = Experiment(self.root, f"{ts}_{safe}")
        exp.write_config(config)
        return exp

    def register(self, exp: Experiment, **fields):
        with _LOCK, self._conn() as c:
            c.execute(
                "INSERT INTO experiments(ts,name,dataset,model,scheme,dir,n_trials,"
                "acc_mean,acc_std,kappa_mean,note) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (fields.get("ts", datetime.now().isoformat(timespec="seconds")),
                 fields.get("name", exp.dir.name), fields.get("dataset", ""),
                 fields.get("model", ""), fields.get("scheme", ""), str(exp.dir),
                 fields.get("n_trials", 0), fields.get("acc_mean"),
                 fields.get("acc_std"), fields.get("kappa_mean"),
                 fields.get("note", "")))

    def list_experiments(self, keyword: str = "") -> list[dict]:
        q = "SELECT * FROM experiments"
        args = ()
        if keyword:
            q += " WHERE name LIKE ? OR model LIKE ? OR dataset LIKE ?"
            args = (f"%{keyword}%",) * 3
        q += " ORDER BY id DESC"
        with self._conn() as c:
            rows = [dict(r) for r in c.execute(q, args).fetchall()]
        # 目录已被删除（如测试清理、手工删除）的残留记录直接过滤掉，
        # 避免界面上出现"点进去报错"的实验；顺手清库。
        alive, dead = [], []
        for r in rows:
            (alive if Path(r["dir"]).is_dir() else dead).append(r)
        if dead:
            with _LOCK, self._conn() as c:
                c.executemany("DELETE FROM experiments WHERE id=?",
                              [(r["id"],) for r in dead])
        return alive

    def find(self, exp_dir: str) -> dict | None:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM experiments WHERE dir=?", (str(exp_dir),)).fetchall()
            return dict(rows[0]) if rows else None

    # ------------------------------------------------------------------
    def delete_experiment(self, exp_dir: str | Path, *, remove_files: bool = True) -> dict:
        """删除一个实验：索引记录 + （可选）整个实验目录。

        Parameters
        ----------
        remove_files : True  → 连同磁盘文件一起删除（不可恢复）
                       False → 只从列表/索引里移除，保留权重与报告

        安全约束：只允许删 ``experiments/`` **根目录之内** 的内容，防止误删
        项目里的其它文件（例如 dir 被写成上级目录或项目根）。

        Returns
        -------
        dict : {dir, removed_dir, removed_rows, freed_bytes}
        """
        d = Path(exp_dir)
        root = self.root.resolve()
        try:
            rel = d.resolve().relative_to(root)
        except ValueError as e:
            raise ValueError(f"拒绝删除：{d} 不在实验目录 {root} 之内") from e
        if not str(rel) or str(rel) == ".":
            raise ValueError(f"拒绝删除实验根目录本身：{root}")

        freed, removed_dir = 0, False
        if remove_files and d.exists():
            freed = dir_size(d)
            shutil.rmtree(d)
            removed_dir = True

        cand = (str(d), str(d.resolve()))
        with _LOCK, self._conn() as c:
            marks = ",".join("?" * len(cand))
            removed_rows = c.execute(
                f"DELETE FROM experiments WHERE dir IN ({marks})", cand).rowcount
        return {"dir": str(d), "removed_dir": removed_dir,
                "removed_rows": removed_rows, "freed_bytes": freed}
