"""Двухуровневая модель.

Уровень 1: бустинг на поведенческих и парных признаках, дает оценку s для каждой куки.
Уровень 2: тот же бустинг плюс агрегаты s соседей из `graph.neighbor_score_features`.

Метка куки не должна попасть в ее же признак через соседей. Поэтому s для куки обучения дает
модель, которая не видела ее двухдневную группу, а для остальных кук - модель на всем обучении.
Одна и та же функция работает и в CV, и в финальном обучении.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier

from data import clean_events, load_events, load_meta
from features import build_behavior_features
from graph import coview_pairs, neighbor_score_features, pair_features

SEED = 42
N_JOBS = int(os.environ.get("N_JOBS", 8))  # на результат не влияет

# Параметры LightGBM из случайного поиска: лучший средний ранг по P@R, PR-AUC и logloss.
# deterministic и force_col_wise дают одинаковый результат при любом числе потоков.
LGB_PARAMS = dict(
    n_estimators=450, learning_rate=0.02, num_leaves=63, min_child_samples=80,
    colsample_bytree=0.5, subsample=1.0, reg_lambda=5.0, reg_alpha=0.1, scale_pos_weight=2.0,
    deterministic=True, force_col_wise=True, verbose=-1, n_jobs=N_JOBS,
)
# XGBoost только для сравнения в ноутбуке, в финальный ансамбль не входит
XGB_PARAMS = dict(
    n_estimators=600, learning_rate=0.02, max_depth=5, min_child_weight=3, subsample=0.8,
    colsample_bytree=0.6, reg_lambda=5.0, tree_method="hist", enable_categorical=True, n_jobs=N_JOBS,
)
# у категориальных признаков до 8 значений, one-hot для них быстрее CTR
CB_PARAMS = dict(
    iterations=800, learning_rate=0.05, depth=6, l2_leaf_reg=5.0, one_hot_max_size=10,
    thread_count=N_JOBS, verbose=0, allow_writing_files=False,
)


class CatBoostWrapper:
    """CatBoost ждет категории строками, приводим их внутри."""

    def __init__(self, seed: int = SEED, **params):
        self.seed, self.params = seed, {**CB_PARAMS, **params}

    def _prep(self, X):
        self.cat_ = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
        return X.astype({c: str for c in self.cat_})

    def fit(self, X, y):
        Xp = self._prep(X)
        self.model_ = CatBoostClassifier(**self.params, random_seed=self.seed, cat_features=self.cat_).fit(Xp, y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X.astype({c: str for c in self.cat_}))


class Blend:
    """Среднее вероятностей нескольких моделей, каждая с несколькими random_state."""

    def __init__(self, makers: list[Callable[[int], object]], seeds=(0, 1, 2)):
        self.makers, self.seeds = makers, seeds

    def fit(self, X, y):
        self.models_ = [make(SEED + s).fit(X, y) for make in self.makers for s in self.seeds]
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models_], axis=0)


def make_lgbm(seed: int = SEED, **overrides) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(**{**LGB_PARAMS, **overrides}, random_state=seed)


def make_xgb(seed: int = SEED, **overrides) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(**{**XGB_PARAMS, **overrides}, random_state=seed)


def make_catboost(seed: int = SEED, **overrides) -> CatBoostWrapper:
    return CatBoostWrapper(seed, **overrides)


def make_final_model() -> Blend:
    """Финальная модель обоих уровней: LightGBM + CatBoost, по 3 сида.

    XGBoost не лучше по качеству, а его прогноз меняется от числа потоков, до 0.2.
    LightGBM и CatBoost дают одинаковый результат на любой машине.
    """
    return Blend([make_lgbm, make_catboost])


def build_dataset(data_dir: str | Path, verbose: bool = True):
    """Загрузка, очистка и признаки для train и test.

    Возвращает meta, X (строки в порядке meta) и pairs (ребра графа).
    """
    meta = load_meta(data_dir)
    ev = clean_events(load_events(data_dir), meta, verbose=verbose)
    pairs = coview_pairs(ev, meta)
    X = build_behavior_features(ev, meta)
    X = X.join(pair_features(pairs, X.index))
    return meta, X, pairs


def window_day(meta: pd.DataFrame) -> np.ndarray:
    """Номер дня окна от начала данных, по нему строим фолды."""
    return (meta.window_start_ts - meta.window_start_ts.min()).dt.days.to_numpy()


def day_groups(days: np.ndarray, size: int = 2) -> list[np.ndarray]:
    u = np.sort(np.unique(days))
    return [u[i:i + size] for i in range(0, len(u), size)]


def stage1_scores(X: pd.DataFrame, y: np.ndarray, train_pos: np.ndarray, day: np.ndarray,
                  make_model: Callable[[], object]) -> pd.Series:
    """Оценки первого уровня для всех строк X без метки самой строки.

    train_pos - позиции строк, на которых можно учиться.
    """
    s = np.full(len(X), np.nan)
    d = day[train_pos]
    for g in day_groups(d):
        hold = train_pos[np.isin(d, g)]
        fit = train_pos[~np.isin(d, g)]
        s[hold] = make_model().fit(X.iloc[fit], y[fit].astype(int)).predict_proba(X.iloc[hold])[:, 1]
    rest = np.setdiff1d(np.arange(len(X)), train_pos)
    s[rest] = make_model().fit(X.iloc[train_pos], y[train_pos].astype(int)).predict_proba(X.iloc[rest])[:, 1]
    return pd.Series(s, index=X.index)


def add_neighbor_features(X: pd.DataFrame, pairs: pd.DataFrame, s: pd.Series) -> pd.DataFrame:
    return X.join(neighbor_score_features(pairs, s, X.index))


def two_stage_fit_predict(X: pd.DataFrame, y: np.ndarray, day: np.ndarray, pairs: pd.DataFrame,
                          train_pos: np.ndarray, pred_pos: np.ndarray,
                          make_model: Callable[[], object]) -> np.ndarray:
    """Учим оба уровня на train_pos, возвращаем прогноз для pred_pos.

    X содержит все куки train и test, соседом может быть любая из них.
    """
    s = stage1_scores(X, y, train_pos, day, make_model)
    X2 = add_neighbor_features(X, pairs, s)
    model = make_model().fit(X2.iloc[train_pos], y[train_pos].astype(int))
    return model.predict_proba(X2.iloc[pred_pos])[:, 1]


def two_stage_fit_predict_fn(X: pd.DataFrame, y: np.ndarray, day: np.ndarray, pairs: pd.DataFrame,
                             make_model: Callable[[], object]):
    """Адаптер к `validation.cross_validate`: переводит срезы train в позиции полной таблицы."""
    def fit_predict(X_fit, _y_fit, X_pred):
        return two_stage_fit_predict(X, y, day, pairs, X.index.get_indexer(X_fit.index),
                                     X.index.get_indexer(X_pred.index), make_model)
    return fit_predict
