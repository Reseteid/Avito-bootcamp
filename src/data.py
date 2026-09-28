"""Загрузка и очистка данных.

Здесь исправляем проблемы сырого лога событий:

1. Полные дубли строк, около 1.5%. Их доля одинакова у ботов и людей, в train и test - это шум
   логирования, удаляем.
2. `platform` пишется по-разному: web, WEB, Web, desktop и так далее. Вариант выбирается
   случайно для каждого события. Приводим к трем значениям.
3. Строки лога идут вперемешку. Сортируем события куки по времени, при равной секунде по коду
   события, это порядок воронки: выдача, карточка, фото, контакт.
4. В train есть ~41 тыс. событий после конца окна, до +16 ч, в test их нет. Среди них все показы
   капчи: кука с капчей после окна - бот в 77% случаев. Это утечка из будущего, поэтому оставляем
   только `window_start_ts <= event_ts < window_end_ts`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PLATFORM_ALIASES = {"desktop": "web", "iphone": "ios"}
META_DATES = ["cookie_created_at", "window_start_ts", "window_end_ts"]


def load_meta(data_dir: str | Path) -> pd.DataFrame:
    """train и test в одной таблице. У test `target` = NaN, колонка `part` показывает источник строки."""
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv", parse_dates=META_DATES)
    test = pd.read_csv(data_dir / "test.csv", parse_dates=META_DATES)
    meta = pd.concat([train.assign(part="train"), test.assign(part="test")], ignore_index=True)
    assert meta.cookie_id.is_unique, "cookie_id должен быть уникален в train+test"
    return meta


def load_events(data_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "events.csv.gz", parse_dates=["event_ts"])


def clean_events(events: pd.DataFrame, meta: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Дедупликация, нормализация платформы, фильтр по окну, сортировка по времени.

    Добавляет колонку `t` - секунды от начала окна.
    """
    n_raw = len(events)
    # дубль - точная копия строки лога
    ev = events.drop_duplicates()
    n_dedup = len(ev)

    ev = ev.assign(platform=ev.platform.str.lower().replace(PLATFORM_ALIASES))

    ev = ev.merge(meta[["cookie_id", "window_start_ts", "window_end_ts"]], on="cookie_id", how="inner")
    in_window = (ev.event_ts >= ev.window_start_ts) & (ev.event_ts < ev.window_end_ts)
    n_before = int((ev.event_ts < ev.window_start_ts).sum())
    n_after = int((ev.event_ts >= ev.window_end_ts).sum())
    ev = ev[in_window]

    ev = ev.assign(t=(ev.event_ts - ev.window_start_ts).dt.total_seconds())
    ev = ev.drop(columns=["window_start_ts", "window_end_ts"])
    ev = ev.sort_values(["cookie_id", "t", "eid"], kind="mergesort").reset_index(drop=True)

    if verbose:
        print(f"событий в файле:          {n_raw:>8,}")
        print(f"  дубликатов:             {n_raw - n_dedup:>8,}")
        print(f"  до начала окна:         {n_before:>8,}")
        print(f"  после конца окна:       {n_after:>8,}  (утечка, отбрасываем)")
        print(f"событий в окнах:          {len(ev):>8,}")
    return ev
