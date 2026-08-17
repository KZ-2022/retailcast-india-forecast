#!/usr/bin/env python3
"""
backtest.py — multi-origin backtest supporting the validation claims in approach_summary.md
(Q5). Truncates sales history at several points before d_1913, forecasts 28 days ahead with the
same hybrid logic as pipeline.py (trailing-window baseline for regime-flagged series, pooled GBM
for the rest), and scores against the truncated actuals using mean RMSSE (scaled by in-sample
naive first-difference RMSE) and WAPE — the same metrics used for final scoring.

Usage:
    python3 backtest.py --data data
"""
import argparse
import os

import numpy as np
import pandas as pd

from gbm_model import prepare_base_frame, forecast_horizon as gbm_forecast_horizon
from pipeline import (
    BLEND_ALPHA,
    EVENT_LIFT_CAP,
    EVENT_LIFT_FALLBACK,
    EVENT_STATE_LIFT,
    FLAT_WINDOW,
    HORIZON,
    REGIME_SERIES_IDS,
    SEASONAL_WINDOW,
    WDAY_SHRINK_N,
    weekday_seasonal_forecast,
)

ORIGINS = [28, 56, 84, 112, 140, 168, 196, 224, 252]  # days back from d_1913 used as backtest cutoffs


def rmsse_wape(pred: np.ndarray, actual: np.ndarray, y_train: np.ndarray):
    scale = np.sqrt(np.mean(np.diff(y_train, axis=1) ** 2, axis=1))
    rmsse_vals = []
    for i in range(actual.shape[0]):
        if scale[i] == 0:
            continue
        rmse = np.sqrt(np.mean((actual[i] - pred[i]) ** 2))
        rmsse_vals.append(rmse / scale[i])
    wape = np.abs(actual - pred).sum() / actual.sum()
    return float(np.mean(rmsse_vals)), float(wape)


def baseline_forecast_from_cut(y_train: np.ndarray, cal: pd.DataFrame, cut: int, states: np.ndarray) -> np.ndarray:
    hist_cal = cal[cal["dnum"] <= cut].sort_values("dnum")
    wday_hist = hist_cal["wday"].to_numpy()
    horizon_cal = cal[(cal["dnum"] > cut) & (cal["dnum"] <= cut + HORIZON)].sort_values("dnum")
    horizon_wday = horizon_cal["wday"].to_numpy()
    horizon_event_name = horizon_cal["event_name_1"].fillna("none").to_numpy()
    horizon_event_type = horizon_cal["event_type_1"].fillna("none").to_numpy()

    p_seasonal = weekday_seasonal_forecast(y_train, SEASONAL_WINDOW, wday_hist, horizon_wday, WDAY_SHRINK_N)
    flat = y_train[:, -FLAT_WINDOW:].mean(axis=1, keepdims=True)
    p_flat = np.tile(flat, (1, HORIZON))
    p = BLEND_ALPHA * p_seasonal + (1 - BLEND_ALPHA) * p_flat

    for j, (ename, etype) in enumerate(zip(horizon_event_name, horizon_event_type)):
        if ename == "none":
            continue
        for i, st in enumerate(states):
            mult = EVENT_STATE_LIFT.get((ename, st))
            if mult is None:
                mult = EVENT_LIFT_FALLBACK.get(etype)
            if mult is not None:
                p[i, j] *= min(mult, EVENT_LIFT_CAP)
    return np.clip(p, 0, None)


def hybrid_forecast_from_cut(sales: pd.DataFrame, cal: pd.DataFrame, sell_prices: pd.DataFrame,
                              y_train: np.ndarray, cut: int, states: np.ndarray, is_regime: np.ndarray) -> np.ndarray:
    p_base = baseline_forecast_from_cut(y_train, cal, cut, states)

    trunc_dcols = pd.DataFrame(y_train, columns=[f"d_{i + 1}" for i in range(cut)])
    sales_trunc = pd.concat(
        [sales[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]].reset_index(drop=True), trunc_dcols],
        axis=1,
    )
    sales_long, cal_price, _ = prepare_base_frame(sales_trunc, cal, sell_prices, n_hist=cut, horizon=HORIZON)
    p_gbm = gbm_forecast_horizon(sales_long, cal_price, cut, n_series=len(sales_trunc))

    return np.where(is_regime[:, None], p_base, p_gbm)


def main(data_dir: str) -> None:
    sales = pd.read_csv(os.path.join(data_dir, "sales_train.csv"))
    cal = pd.read_csv(os.path.join(data_dir, "calendar.csv"))
    cal["dnum"] = cal["d"].str.replace("d_", "", regex=False).astype(int)
    sell_prices = pd.read_csv(os.path.join(data_dir, "sell_prices.csv"))

    n_hist = 1913
    dcols = [f"d_{i + 1}" for i in range(n_hist)]
    y_all = sales[dcols].to_numpy(float)
    states = sales["state_id"].to_numpy()
    is_regime = sales["id"].isin(REGIME_SERIES_IDS).to_numpy()

    print(f"{'cut':>6} {'model_RMSSE':>12} {'model_WAPE':>11} {'naive_RMSSE':>12} {'naive_WAPE':>11}")
    per_origin_rmsse, per_origin_wape = [], []
    for days_back in ORIGINS:
        cut = n_hist - days_back
        y_train = y_all[:, :cut]
        actual = y_all[:, cut:cut + HORIZON]

        pred = hybrid_forecast_from_cut(sales, cal, sell_prices, y_train, cut, states, is_regime)
        r, w = rmsse_wape(pred, actual, y_train)
        per_origin_rmsse.append(r)
        per_origin_wape.append(w)

        naive = np.tile(y_train[:, -7:], (1, 4))[:, :HORIZON]
        rn, wn = rmsse_wape(naive, actual, y_train)

        print(f"{cut:>6} {r:>12.3f} {w:>11.3f} {rn:>12.3f} {wn:>11.3f}")

    print()
    print(f"mean model RMSSE: {np.mean(per_origin_rmsse):.4f}")
    print(f"mean model WAPE:  {np.mean(per_origin_wape):.4f}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(here, "data"))
    args = ap.parse_args()
    main(args.data)

