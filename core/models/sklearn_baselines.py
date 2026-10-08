"""传统机器学习基线：CSP/FBCSP + LDA/SVM（sklearn 接口，与 PyTorch 路径统一）。"""
from __future__ import annotations

import numpy as np


class ClassicBaseline:
    """统一 fit/predict 接口的经典分类器。

    kind:
        csp_lda    : CSP + LDA
        csp_svm    : CSP + 线性 SVM
        fbcsp_lda  : 滤波器组 CSP + LDA（4-40 Hz，每 4Hz 一个带）
    """

    def __init__(self, kind: str = "csp_lda", n_components: int = 6, C: float = 1.0):
        self.kind = kind
        self.n_components = int(n_components)
        self.C = float(C)

    # ------------------------------------------------------------------
    def _make_csp(self):
        from mne.decoding import CSP
        return CSP(n_components=self.n_components, reg="ledoit_wolf", log=True)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        if self.kind in ("csp_lda", "csp_svm"):
            from mne.decoding import CSP
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            from sklearn.pipeline import Pipeline
            from sklearn.svm import SVC

            clf = (LinearDiscriminantAnalysis(solver="svd") if self.kind == "csp_lda"
                   else SVC(C=self.C, kernel="linear"))
            self.pipe_ = Pipeline([("csp", self._make_csp()), ("clf", clf)])
            self.pipe_.fit(X, y)
        elif self.kind == "fbcsp_lda":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

            self.bands_ = [(f, f + 4) for f in range(4, 40, 4)]
            from mne.decoding import CSP
            self.csps_ = [CSP(n_components=self.n_components, reg="ledoit_wolf", log=True)
                          for _ in self.bands_]
            feats = np.concatenate([csp.fit_transform(X, y) for csp in self.csps_], axis=1)
            # 每个 CSP 分别 fit（与论文常见做法一致：滤波器组各自学习）
            self.clf_ = LinearDiscriminantAnalysis(solver="svd")
            self.clf_.fit(feats, y)
        else:
            raise ValueError(f"未知经典基线：{self.kind}")
        return self

    def _features(self, X: np.ndarray) -> np.ndarray:
        if self.kind in ("csp_lda", "csp_svm"):
            return self.pipe_.transform(X)
        return np.concatenate(
            [csp.transform(X) for csp, (f1, f2) in zip(self.csps_, self.bands_)], axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.kind in ("csp_lda", "csp_svm"):
            return self.pipe_.predict(X)
        return self.clf_.predict(self._features(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率（列顺序与 ``classes_`` 一致，供 AUC / Brier / ECE 使用）。

        - CSP+LDA / FBCSP+LDA：直接转发 ``predict_proba``
        - CSP+SVM（线性核、未开 probability）：用 decision_function 做 softmax 近似
        """
        X = np.asarray(X, dtype=np.float64)
        if self.kind in ("csp_lda", "csp_svm"):
            # 必须走整条 pipeline（CSP → 分类器），不能直接喂原始 X
            if hasattr(self.pipe_, "predict_proba"):
                return self.pipe_.predict_proba(X)
            d = np.asarray(self.pipe_.decision_function(X), dtype=float)
            if d.ndim == 1:                 # 二分类 → (N, 2)
                d = np.c_[-d, d]
            d = d - d.max(axis=1, keepdims=True)
            e = np.exp(d)
            return e / e.sum(axis=1, keepdims=True)
        return self.clf_.predict_proba(self._features(X))   # fbcsp：特征已变换

    @property
    def classes_(self):
        est = self.pipe_[-1] if self.kind in ("csp_lda", "csp_svm") else self.clf_
        return getattr(est, "classes_", None)

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y)).mean())


def is_classic(name: str) -> bool:
    return name in ("csp_lda", "csp_svm", "fbcsp_lda")
