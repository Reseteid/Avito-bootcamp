"""Поведенческие признаки куки.

На вход идут события после `data.clean_events`. Каждый блок возвращает DataFrame с индексом
`cookie_id` и берет только события самой куки, поэтому признак известен к концу окна.
Признаки по другим кукам лежат в `graph.py`.

Пропуски: если события не было, счетчик равен 0. Если у доли нулевой знаменатель (например,
глубина листания у куки без поиска), оставляем NaN - бустинг обрабатывает его сам.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

EVENT_TYPES = [
    "search_results_view", "item_view", "photo_swipe", "seller_page_view",
    "contact_phone_show", "contact_chat_open", "contact_message_sent",
    "favorite_add", "login",
]
SESSION_GAP_S = 30 * 60      # пауза больше 30 минут начинает новую сессию
SCREEN_W, SCREEN_H = 1920, 1080  # размер экрана в координатах курсора


# ----------------------------------------------------------------------------- helpers

def _entropy(counts: pd.Series, level: str = "cookie_id") -> pd.Series:
    """Энтропия Шеннона. Индекс counts: [cookie_id, значение]."""
    p = counts / counts.groupby(level=level).transform("sum")
    return -(p * np.log(p)).groupby(level=level).sum()


def _safe_div(a, b):
    return a / b.where(b > 0)


# ----------------------------------------------------------------------------- blocks

def meta_features(meta: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """Возраст куки. Среди кук моложе 3 суток ботов 16-18%, среди старых ~5%."""
    m = meta.set_index("cookie_id")
    age_h = (m.window_start_ts - m.cookie_created_at).dt.total_seconds() / 3600
    first_t = ev.groupby("cookie_id").t.min()
    return pd.DataFrame({
        "cookie_age_h": age_h,
        "log_cookie_age_h": np.log1p(age_h),
        # возраст куки в момент первого события окна
        "age_at_first_event_h": age_h + first_t.reindex(m.index) / 3600,
    })


def volume_features(ev: pd.DataFrame) -> pd.DataFrame:
    """Сколько событий каждого типа и их доли."""
    cnt = pd.crosstab(ev.cookie_id, ev.event_name).reindex(columns=EVENT_TYPES, fill_value=0)
    n = cnt.sum(axis=1)
    f = pd.DataFrame({"n_events": n, "log_n_events": np.log1p(n)})
    for e in EVENT_TYPES:
        f[f"cnt_{e}"] = cnt[e]
        f[f"share_{e}"] = cnt[e] / n
    contacts = cnt[["contact_phone_show", "contact_chat_open", "contact_message_sent"]].sum(axis=1)
    f["share_contacts"] = contacts / n
    # фото, избранное и логин чаще у людей
    f["photo_per_item_view"] = _safe_div(cnt.photo_swipe, cnt.item_view)
    f["n_event_types"] = (cnt > 0).sum(axis=1)
    f["event_type_entropy"] = _entropy(ev.groupby(["cookie_id", "event_name"]).size())
    return f


def item_features(ev: pd.DataFrame) -> pd.DataFrame:
    """Сколько разных объявлений, категорий, городов и типов продавцов смотрит кука."""
    it = ev[ev.item_id.notna()]
    g = it.groupby("cookie_id")
    n_item_ev = g.size()
    n_items = g.item_id.nunique()
    views_per_item = it.groupby(["cookie_id", "item_id"]).size()

    f = pd.DataFrame({
        "n_items": n_items,
        "item_revisit_rate": 1 - n_items / n_item_ev,
        "max_views_per_item": views_per_item.groupby("cookie_id").max(),
        "share_seller_pro": (it.seller_type == "pro").groupby(it.cookie_id).sum() / n_item_ev,
    })
    # категории и города берем по всем событиям: у выдачи они тоже есть
    ga = ev.groupby("cookie_id")
    n = ga.size()
    f["n_categories"] = ga.item_category.nunique().reindex(n.index)
    f["n_locations"] = ga.item_location.nunique().reindex(n.index)
    f["locations_per_event"] = f.n_locations / n
    f["category_entropy"] = _entropy(ev.groupby(["cookie_id", "item_category"]).size())
    f["location_entropy"] = _entropy(ev.groupby(["cookie_id", "item_location"]).size())
    top_cat = ev.groupby(["cookie_id", "item_category"]).size().groupby("cookie_id").max()
    f["top_category_share"] = top_cat / ga.item_category.count()
    # доля пропусков в атрибутах объявления
    f["share_missing_category"] = it.item_category.isna().groupby(it.cookie_id).mean()
    f["share_missing_location"] = ev.item_location.isna().groupby(ev.cookie_id).mean()
    return f


def search_features(ev: pd.DataFrame) -> pd.DataFrame:
    """Поиск: глубина листания, повторы запроса, переход на следующую страницу того же запроса."""
    s = ev[ev.event_name == "search_results_view"].copy()
    g = s.groupby("cookie_id")
    n_s = g.size()
    prev_q = g.search_query.shift()
    prev_p = g.search_page.shift()
    has_prev = prev_q.notna()
    s["seq_page"] = (s.search_query == prev_q) & (s.search_page == prev_p + 1)
    s["same_page"] = (s.search_query == prev_q) & (s.search_page == prev_p)
    s["has_prev"] = has_prev
    g = s.groupby("cookie_id")
    n_pairs = g.has_prev.sum()

    return pd.DataFrame({
        "n_queries": g.search_query.nunique(),
        "query_repeat_rate": 1 - g.search_query.nunique() / n_s,
        "page_max": g.search_page.max(),
        "page_mean": g.search_page.mean(),
        "page_median": g.search_page.median(),
        "share_page1": (s.search_page == 1).groupby(s.cookie_id).mean(),
        "share_page_ge5": (s.search_page >= 5).groupby(s.cookie_id).mean(),
        "share_seq_pages": _safe_div(g.seq_page.sum(), n_pairs),
        "share_same_page": _safe_div(g.same_page.sum(), n_pairs),
        "n_search_locations": g.item_location.nunique(),
    })


def time_features(ev: pd.DataFrame) -> pd.DataFrame:
    """Паузы между соседними событиями, их регулярность, сессии и время суток."""
    t = ev.t
    cid = ev.cookie_id
    first_in_cookie = cid.ne(cid.shift())
    dt = t.diff().mask(first_in_cookie)  # пауза перед событием, у первого события NaN
    d = pd.DataFrame({"cookie_id": cid, "dt": dt, "t": t}).dropna(subset=["dt"])
    d["log_dt"] = np.log1p(d.dt)
    g = d.groupby("cookie_id")

    f = pd.DataFrame({
        "gap_mean": g.dt.mean(),
        "gap_std": g.dt.std(),
        "gap_min": g.dt.min(),
        "gap_max": g.dt.max(),
        "gap_q10": g.dt.quantile(0.10),
        "gap_q25": g.dt.quantile(0.25),
        "gap_median": g.dt.median(),
        "gap_q75": g.dt.quantile(0.75),
        "gap_q90": g.dt.quantile(0.90),
        "log_gap_mean": g.log_dt.mean(),
        "log_gap_std": g.log_dt.std(),
    })
    f["gap_cv"] = f.gap_std / f.gap_mean
    # burstiness: -1 для периодичного потока, 0 для пуассоновского, 1 для всплесков
    f["burstiness"] = (f.gap_std - f.gap_mean) / (f.gap_std + f.gap_mean)

    # доли пауз по диапазонам: у ботов много пауз 3-15 с, у людей паузы разные
    bins = [(-1, 1), (1, 3), (3, 8), (8, 15), (15, 30), (30, 60), (60, 300), (300, SESSION_GAP_S)]
    for lo, hi in bins:
        f[f"share_gap_{lo + 1}_{hi}s"] = ((d.dt > lo) & (d.dt <= hi)).groupby(d.cookie_id).mean()
    f["share_gap_gt30m"] = (d.dt > SESSION_GAP_S).groupby(d.cookie_id).mean()

    # паузы внутри сессий, без перерывов между сессиями
    ins = d[d.dt <= SESSION_GAP_S].groupby("cookie_id").dt
    f["in_session_gap_mean"] = ins.mean()
    f["in_session_gap_median"] = ins.median()
    f["in_session_gap_std"] = ins.std()
    f["in_session_gap_cv"] = f.in_session_gap_std / f.in_session_gap_mean

    # local variation: ~1 для пуассоновского потока, ~0 для периодичного
    nxt = d.groupby("cookie_id").dt.shift(-1)
    lv = 3 * ((d.dt - nxt) / (d.dt + nxt)) ** 2
    f["local_variation"] = lv.groupby(d.cookie_id).mean()

    # сессии и активность по часам
    ga = ev.groupby("cookie_id")
    n = ga.size()
    session_id = (dt.isna() | (dt > SESSION_GAP_S)).groupby(cid).cumsum()
    sess = pd.DataFrame({"cookie_id": cid, "sid": session_id, "t": t}).groupby(["cookie_id", "sid"]).t
    sess_len = sess.size()
    sess_dur = sess.max() - sess.min()
    f = f.reindex(n.index)
    f["n_sessions"] = sess_len.groupby("cookie_id").size()
    f["events_per_session"] = n / f.n_sessions
    f["max_session_events"] = sess_len.groupby("cookie_id").max()
    f["mean_session_dur"] = sess_dur.groupby("cookie_id").mean()
    f["max_session_dur"] = sess_dur.groupby("cookie_id").max()
    f["active_span"] = ga.t.max() - ga.t.min()
    f["events_per_active_min"] = n / (f.active_span / 60 + 1)
    f["first_event_hour"] = ga.t.min() / 3600
    f["last_event_hour"] = ga.t.max() / 3600

    hour = (ev.t // 3600).astype(int)
    hc = pd.DataFrame({"cookie_id": cid, "h": hour}).groupby(["cookie_id", "h"]).size()
    f["n_active_hours"] = hc.groupby("cookie_id").size()
    f["hour_entropy"] = _entropy(hc)
    f["share_night"] = (hour < 6).groupby(cid).mean()
    f["share_same_second"] = (dt == 0).groupby(cid).sum() / n
    return f


def dwell_features(ev: pd.DataFrame) -> pd.DataFrame:
    """Время чтения: пауза после события до следующего события куки.

    Человек читает карточку или выдачу ~45 с, парсер ~13 с. Перерывы между сессиями не берем.
    """
    cid = ev.cookie_id
    last_in_cookie = cid.ne(cid.shift(-1))
    dwell = (ev.t.shift(-1) - ev.t).mask(last_in_cookie)
    d = pd.DataFrame({"cookie_id": cid, "e": ev.event_name, "dwell": dwell})
    d = d[d.dwell.notna() & (d.dwell <= SESSION_GAP_S)]
    f = pd.DataFrame(index=cid.unique())
    for e, name in [("item_view", "item"), ("search_results_view", "search"), ("photo_swipe", "photo")]:
        x = d[d.e == e].groupby("cookie_id").dwell
        f[f"dwell_{name}_median"] = x.median()
        f[f"dwell_{name}_mean"] = x.mean()
        f[f"dwell_{name}_std"] = x.std()
    return f


def transition_features(ev: pd.DataFrame) -> pd.DataFrame:
    """Доли переходов между типами событий."""
    cid = ev.cookie_id
    same_cookie = cid.eq(cid.shift())
    prev_e = ev.event_name.shift().where(same_cookie)
    e = ev.event_name
    n_tr = same_cookie.groupby(cid).sum()
    pairs = {
        "search_to_item": (prev_e == "search_results_view") & (e == "item_view"),
        "search_to_search": (prev_e == "search_results_view") & (e == "search_results_view"),
        "item_to_item": (prev_e == "item_view") & (e == "item_view"),
        "item_to_photo": (prev_e == "item_view") & (e == "photo_swipe"),
        "item_to_seller": (prev_e == "item_view") & (e == "seller_page_view"),
        "same_type": same_cookie & (prev_e == e),
    }
    f = pd.DataFrame(index=n_tr.index)
    for k, mask in pairs.items():
        f[f"tr_{k}"] = _safe_div(mask.groupby(cid).sum(), n_tr)

    # действие с объявлением (фото, контакт, избранное) до открытия его карточки;
    # человек обычно сначала открывает карточку
    first_view = (ev[ev.event_name == "item_view"].groupby(["cookie_id", "item_id"]).t.min()
                  .rename("first_view_t").reset_index())
    acts = ev[ev.item_id.notna() & (ev.event_name != "item_view")]
    acts = acts[["cookie_id", "item_id", "t"]].merge(first_view, on=["cookie_id", "item_id"], how="left")
    no_view = ~(acts.first_view_t <= acts.t)  # карточку не открывал вовсе: тоже True
    f["share_action_without_view"] = no_view.groupby(acts.cookie_id).mean()
    return f


def pointer_features(ev: pd.DataFrame) -> pd.DataFrame:
    """Курсор, только web.

    У людей курсор равномерно покрывает экран, std по x около 550. У ботов он собран у одной
    точки, std около 200, и чаще упирается в край. Координаты у ботов есть вдвое реже.
    """
    web = ev[ev.platform == "web"]
    p = web[web.pointer_x.notna()].copy()
    p["edge"] = ((p.pointer_x <= 0) | (p.pointer_x >= SCREEN_W) |
                 (p.pointer_y <= 0) | (p.pointer_y >= SCREEN_H))
    p["dist_center"] = np.hypot(p.pointer_x - SCREEN_W / 2, p.pointer_y - SCREEN_H / 2)
    same = p.cookie_id.eq(p.cookie_id.shift())
    p["jump"] = np.hypot(p.pointer_x.diff(), p.pointer_y.diff()).where(same)
    gp = p.groupby("cookie_id")
    f = pd.DataFrame({
        "pointer_share": web.pointer_x.notna().groupby(web.cookie_id).mean(),
        "pointer_n": gp.size(),
        "pointer_x_mean": gp.pointer_x.mean(),
        "pointer_y_mean": gp.pointer_y.mean(),
        "pointer_x_std": gp.pointer_x.std(),
        "pointer_y_std": gp.pointer_y.std(),
        "pointer_edge_share": gp.edge.mean(),
        "pointer_center_dist": gp.dist_center.mean(),
        "pointer_jump_mean": gp.jump.mean(),
    })
    f.loc[f.pointer_n.isna(), "pointer_n"] = 0
    return f


# ----------------------------------------------------------------------------- user agent

_SCRIPT_UA = re.compile(r"^(python-|curl/|Go-http-client|Scrapy|node-fetch|okhttp/|Java/|wget)", re.I)


def ua_family(ua: str) -> str:
    """Тип клиента по строке User-Agent.

    Сырая строка дает 148 значений с версиями и моделями устройств, на них модель
    переобучается. Берем только тип клиента.
    """
    if "HeadlessChrome" in ua:
        return "headless"
    if _SCRIPT_UA.match(ua):
        return "http_lib"
    if ua.startswith("Avito/"):
        return "app"
    if "YaBrowser" in ua:
        return "yabrowser"
    if "Firefox" in ua:
        return "firefox"
    if "iPhone" in ua:
        return "mobile_safari"
    if "Android" in ua:
        return "mobile_chrome"
    return "desktop_chrome"


def ua_features(ev: pd.DataFrame) -> pd.DataFrame:
    fam = ev.user_agent.map(ua_family)
    cid = ev.cookie_id
    fam_counts = fam.groupby([cid, fam]).size()
    main = fam_counts.groupby(level=0).idxmax().str[1]  # самый частый тип клиента у куки
    return pd.DataFrame({
        "ua_family": main,
        "ua_automation": fam.isin(["headless", "http_lib"]).groupby(cid).mean(),
        "n_user_agents": ev.user_agent.groupby(cid).nunique(),
        "n_ua_families": fam.groupby(cid).nunique(),
    })


def platform_feature(ev: pd.DataFrame) -> pd.Series:
    return ev.groupby("cookie_id").platform.first().rename("platform")


# ----------------------------------------------------------------------------- assembly

CATEGORICAL = ["platform", "ua_family"]


def feature_blocks(ev: pd.DataFrame, meta: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Блоки признаков по смыслу."""
    return {
        "возраст куки": meta_features(meta, ev),
        "объем и состав событий": volume_features(ev),
        "объявления, категории, города": item_features(ev),
        "поиск и листание выдачи": search_features(ev),
        "паузы, сессии, время суток": time_features(ev),
        "время чтения страниц": dwell_features(ev),
        "переходы между событиями": transition_features(ev),
        "курсор": pointer_features(ev),
        "user agent и платформа": pd.concat([ua_features(ev), platform_feature(ev)], axis=1),
    }


def build_behavior_features(ev: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Все поведенческие признаки, по строке на куку в порядке meta."""
    X = pd.concat(feature_blocks(ev, meta).values(), axis=1).reindex(meta.cookie_id)
    X.index.name = "cookie_id"
    # нет событий - счетчик 0
    cnt_cols = [c for c in X.columns if c.startswith("cnt_")] + ["n_items", "n_queries", "pointer_n"]
    X[cnt_cols] = X[cnt_cols].fillna(0)
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    return X
