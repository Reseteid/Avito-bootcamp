"""Собирает submission.csv с нуля.

    python make_submission.py
    python make_submission.py --data path/to/data --out my_submission.csv

Очистка событий, признаки, двухуровневая модель на всем train, score для кук из test.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline import build_dataset, make_final_model, two_stage_fit_predict, window_day  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data"))
    parser.add_argument("--out", default=str(ROOT / "submission.csv"))
    args = parser.parse_args()

    meta, X, pairs = build_dataset(args.data)
    y = meta.target.to_numpy()
    train_pos = np.flatnonzero(meta.part.eq("train"))
    test_pos = np.flatnonzero(meta.part.eq("test"))

    score = two_stage_fit_predict(X, y, window_day(meta), pairs, train_pos, test_pos, make_final_model)

    sub = pd.DataFrame({"cookie_id": meta.cookie_id.iloc[test_pos].to_numpy(), "score": score})
    test_ids = pd.read_csv(Path(args.data) / "test.csv", usecols=["cookie_id"]).cookie_id
    assert len(sub) == len(test_ids) and set(sub.cookie_id) == set(test_ids), "cookie_id не совпадают с test.csv"
    assert sub.cookie_id.is_unique and sub.score.notna().all() and sub.score.between(0, 1).all()
    sub.to_csv(args.out, index=False)
    print(f"{args.out}: {len(sub)} строк")


if __name__ == "__main__":
    main()
