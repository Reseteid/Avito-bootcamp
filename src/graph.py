"""Граф общих объявлений.

Люди выбирают объявления почти равномерно из ~60 тыс., боты - из узкого списка своего сервиса,
и этот список со временем почти не меняется. Если две куки за неделю открыли 3+ общих объявления,
это почти всегда боты одного сервиса. Общие просмотры с людьми сигнала не дают.

Время: куку дня d связываем только с куками дней [d-L+1, d]. Окна всех кук дня совпадают,
поэтому к концу окна эти события уже есть. Будущие дни не берем.

Два уровня признаков:
* `pair_features` - без меток: сколько объявлений кука делит с соседями;
* `neighbor_score_features` - оценки соседей от модели первого уровня. Модель для этих оценок не
  видела меток тех кук, для которых строим признак. Подробности в `pipeline.py`.

Счетчики "сколько кук открывали то же объявление" не берем: число кук по дням разное, от 606 до 912,
в неделю перед тестом их меньше, поэтому такие счетчики сдвинуты между train и test, а нормировка
на размер дня позволяет модели узнать день. Прироста они не дают. По той же причине не берем
максимум оценки по всем соседям: он растет с числом соседей.
"""

from __future__ import annotations

import pandas as pd

LOOKBACK_DAYS = 7  # текущий день и 6 прошлых


def _cookie_day(meta: pd.DataFrame) -> pd.Series:
    """Номер дня окна от начала данных, день 0 - 2026-04-06."""
    day0 = meta.window_start_ts.min()
    return meta.set_index("cookie_id").window_start_ts.sub(day0).dt.days


def coview_pairs(ev: pd.DataFrame, meta: pd.DataFrame, lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Ребра графа: (cookie_id, neighbor_id, n_shared).

    neighbor - кука из дней [d-L+1, d], у нее n_shared общих объявлений с cookie_id.
    Связь идет только назад во времени.
    """
    v = ev.loc[ev.item_id.notna(), ["cookie_id", "item_id"]].drop_duplicates()
    v["day"] = v.cookie_id.map(_cookie_day(meta)).to_numpy()
    pairs = v.merge(v, on="item_id", suffixes=("", "_nb"))
    pairs = pairs[(pairs.cookie_id != pairs.cookie_id_nb)
                  & (pairs.day_nb <= pairs.day) & (pairs.day_nb > pairs.day - lookback_days)]
    return (pairs.groupby(["cookie_id", "cookie_id_nb"]).size().rename("n_shared").reset_index()
            .rename(columns={"cookie_id_nb": "neighbor_id"}))


def pair_features(pairs: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """Без меток: максимум общих объявлений с одним соседом и число соседей с 2+ общими.
    У людей 2+ общих объявления почти не бывает."""
    g = pairs.groupby("cookie_id").n_shared
    f = pd.DataFrame({
        "pair_max_shared": g.max(),
        "pair_n_share2": (pairs.n_shared >= 2).groupby(pairs.cookie_id).sum(),
    })
    return f.reindex(index).fillna(0)  # нет общих объявлений - 0


def neighbor_score_features(pairs: pd.DataFrame, score: pd.Series, index: pd.Index) -> pd.DataFrame:
    """Агрегаты оценок соседей.

    score - оценка модели первого уровня для каждой куки. Сумма оценок соседей примерно равна
    числу ботов среди них. Соседей с 2+ общими объявлениями считаем отдельно.
    """
    e = pairs.assign(s=pairs.neighbor_id.map(score).to_numpy())
    e["ws"] = e.n_shared * e.s
    strong = e[e.n_shared >= 2]
    g, gs = e.groupby("cookie_id"), strong.groupby("cookie_id")
    f = pd.DataFrame({
        "nb_sum_score": g.s.sum(),
        "nb_wsum_score": g.ws.sum(),
        "nb2_max_score": gs.s.max(),
        "nb2_sum_score": gs.s.sum(),
    }).reindex(index)
    sums = ["nb_sum_score", "nb_wsum_score", "nb2_sum_score"]
    f[sums] = f[sums].fillna(0)  # нет соседей - сумма 0, максимум остается NaN
    return f
