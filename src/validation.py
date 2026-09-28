"""Схемы валидации и метрики.

Метрика из официального `metric.py`. Своя реализация легко разойдется с ней на равных score.

Train - 14 суточных окон с 6 по 19 апреля, фолды строим по дням:
* forward - основная схема: учим только на прошлых днях, 4 фолда по 2 дня валидации;
* by_day - 7 фолдов по 2 дня, учим на остальных днях. Дрейфа между днями нет, adversarial
  AUC около 0.5. Зато прогноз вне обучения есть для всех 899 ботов, и сравнения меньше шумят.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metric import precision_at_recall, recall_at_fpr  # noqa: E402  (официальная метрика)

FitPredict = Callable[[pd.DataFrame, np.ndarray, pd.DataFrame], np.ndarray]


def score_all(y, p) -> dict:
    return {
        "P@R0.7": precision_at_recall(y, p),
        "PR-AUC": average_precision_score(y, p),
        "ROC-AUC": roc_auc_score(y, p),
        "R@FPR1%": recall_at_fpr(y, p, 0.01),
    }


def forward_folds(day: pd.Series, n_val_days: int = 2, n_folds: int = 4):
    """Расширяющееся окно: валидация на последних n_folds блоках по n_val_days дней."""
    days = np.sort(day.unique())
    folds = []
    for k in range(n_folds, 0, -1):
        val_days = days[len(days) - k * n_val_days: len(days) - (k - 1) * n_val_days]
        tr = np.where(day < val_days.min())[0]
        va = np.where(day.isin(val_days))[0]
        folds.append((tr, va))
    return folds


def by_day_folds(day: pd.Series, n_val_days: int = 2):
    days = np.sort(day.unique())
    folds = []
    for i in range(0, len(days), n_val_days):
        val_days = days[i:i + n_val_days]
        va = np.where(day.isin(val_days))[0]
        tr = np.where(~day.isin(val_days))[0]
        folds.append((tr, va))
    return folds


@dataclass
class CVResult:
    name: str
    oof: np.ndarray                     # NaN, если кука не попала в валидацию
    fold_scores: pd.DataFrame
    pooled: dict = field(default_factory=dict)

    def summary(self) -> dict:
        mean = self.fold_scores.mean()
        std = self.fold_scores.std()
        out = {"model": self.name}
        out["P@R0.7 folds mean"] = mean["P@R0.7"]
        out["P@R0.7 folds std"] = std["P@R0.7"]
        for k, v in self.pooled.items():
            out[f"{k} pooled"] = v
        return out


def cross_validate(fit_predict: FitPredict, X: pd.DataFrame, y: np.ndarray, folds, name: str = "") -> CVResult:
    oof = np.full(len(y), np.nan)
    rows = []
    for tr, va in folds:
        p = fit_predict(X.iloc[tr], y[tr], X.iloc[va])
        oof[va] = p
        rows.append(score_all(y[va], p))
    done = ~np.isnan(oof)
    return CVResult(name, oof, pd.DataFrame(rows), score_all(y[done], oof[done]))


def cached(path: str | Path, compute: Callable[[], object]):
    """Кеш долгих экспериментов в artifacts/. Удалите файл, чтобы пересчитать."""
    path = Path(path)
    if path.exists():
        return pd.read_pickle(path)
    result = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(result, path)
    return result


def bootstrap_ci(y, p, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Бутстреп-интервал P@R0.7 по кукам, чтобы видеть шум метрики."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        vals.append(precision_at_recall(y[idx], p[idx]))
    return np.quantile(vals, [alpha / 2, 1 - alpha / 2])
