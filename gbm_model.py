#!/usr/bin/env python3
"""
gbm_model.py -- pooled gradient-boosted-tree forecaster for the 60 RetailCast product-store series.

Used by pipeline.py ONLY for the 45 series with no evidence of a regime change (see
REGIME_SERIES_IDS in pipeline.py) -- pooling across series with a proven ramp/decline risks
re-learning the "old regime" as signal for exactly those series, so the trailing-window
per-series baseline is kept for them instead (see approach_summary.md Q4 for the backtest that
justified this split: hybrid mean RMSSE 0.888 beats both the pure baseline, 0.893, and the pure
pooled model, 0.891).

Design (a pooled direct-multistep gradient-boosting forecaster, scaled down for 60 series):

- Data is reshaped from wide (d_1..d_N columns) to long (one row per series x day).
- Two kinds of features, kept strictly separate so nothing ever looks past its rightful cutoff:
    (a) "anchor" features -- derived purely from past sales, looked up at day (target_day - h):
        lag 7/14/28, rolling mean/std over 7/14/28 days (computed on data strictly before the
        anchor day itself, via an extra shift), and static per-series categorical codes
        (item_id, dept_id, cat_id, store_id, state_id).
    (b) "target-day" features -- properties of the day being predicted, which in this dataset's
        setup are legitimately known in advance (calendar.csv and sell_prices.csv both cover the
        forecast horizon): weekday, month, days-since/until nearest event, state SNAP flag,
        sell_price, and week-over-week sell_price pct-change.
  Feature (b) values are looked up at the *target* day, never the anchor day -- e.g. weekday
  must describe the day being forecast, not the day 7/14/28 days before it.
- Multi-step horizon (28 days) uses a direct multi-step scheme: one pooled model per horizon
  bucket (h in {1-7, 8-14, 15-21, 22-28}) with `h` (steps-ahead) as an explicit feature, trained
  across all series and all valid anchor days at once. This avoids recursive-forecast error
  accumulation while keeping the number of trained models small (4 per origin) given only 60
  underlying series.
- Objective: tweedie (count-appropriate; handles the ~40% zero-inflated series in this dataset
  without a separate zero-inflation model).
- Heavy regularization: shallow trees (max_depth=3), high min_child_weight, small max_leaves,
  strong L1/L2, row/col subsampling, and early stopping on a held-out trailing window -- all to
  guard against overfitting given only 60 pooled series feeding the model.
"""
import numpy as np
import pandas as pd
import xgboost as xgb

HORIZON = 28
LAGS = [7, 14, 28]
ROLL_WINDOWS = [7, 14, 28]
MAX_LAG = max(LAGS)
# Horizon buckets for the direct multi-step scheme.
H_BUCKETS = [(1, 7), (8, 14), (15, 21), (22, 28)]

XGB_PARAMS = dict(
    objective="reg:tweedie",
    tweedie_variance_power=1.2,
    max_depth=3,
    min_child_weight=20,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=1.0,
    reg_lambda=5.0,
    max_leaves=8,
    grow_policy="lossguide",
    n_jobs=4,
    verbosity=0,
)
N_ESTIMATORS_MAX = 500
EARLY_STOPPING_ROUNDS = 30

CAT_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
ANCHOR_COLS = (
    [f"lag_{l}" for l in LAGS]
    + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
    + [f"roll_std_{w}" for w in ROLL_WINDOWS]
    + [f"{c}_code" for c in CAT_COLS]
)
TARGET_DAY_COLS = [
    "wday", "month", "days_since_event", "days_until_event", "snap",
    "sell_price", "price_pct_change",
]
FEATURE_COLS = ANCHOR_COLS + TARGET_DAY_COLS + ["h"]


def _dnum(colname: str) -> int:
    return int(colname.split("_")[1])


def build_calendar_features(cal: pd.DataFrame) -> pd.DataFrame:
    """Adds days-since-event, days-until-event, snap-by-state columns. `cal` may (and for this
    module's use, always does) extend past the training cutoff -- calendar.csv covers the full
    forecast horizon in advance, which is not target leakage."""
    cal = cal.sort_values("dnum").reset_index(drop=True).copy()
    has_event = cal["event_name_1"].notna() | cal["event_name_2"].notna()

    last_event_day = np.where(has_event, cal["dnum"], np.nan)
    last_event_day = pd.Series(last_event_day).ffill()
    cal["days_since_event"] = (cal["dnum"] - last_event_day).fillna(9999)

    next_event_day = np.where(has_event, cal["dnum"], np.nan)
    next_event_day = pd.Series(next_event_day).bfill()
    cal["days_until_event"] = (next_event_day - cal["dnum"]).fillna(9999)

    return cal


def wide_to_long(sales: pd.DataFrame, n_hist: int) -> pd.DataFrame:
    """Melt d_1..d_n_hist columns into long format: one row per (series, dnum).

    series_idx is assigned explicitly from the row order of `sales` (NOT via
    groupby(...).ngroup(), which would reorder series alphabetically by id and silently break
    alignment with any array -- e.g. y_all in the backtest script -- indexed by the original
    sales-dataframe row order).
    """
    dcols = [f"d_{i + 1}" for i in range(n_hist)]
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    sales = sales.reset_index(drop=True).copy()
    sales["series_idx"] = np.arange(len(sales))
    long_df = sales[id_cols + dcols + ["series_idx"]].melt(
        id_vars=id_cols + ["series_idx"], value_vars=dcols, var_name="d", value_name="y"
    )
    long_df["dnum"] = long_df["d"].map(_dnum)
    long_df = long_df.drop(columns=["d"])
    return long_df.sort_values(["series_idx", "dnum"]).reset_index(drop=True)


def add_lag_roll_features(sales_long: pd.DataFrame) -> pd.DataFrame:
    """Add lag-N and rolling mean/std features on the sales-only long frame, computed strictly
    on past y within each series. These become the "anchor" features looked up at dnum."""
    sales_long = sales_long.sort_values(["series_idx", "dnum"]).reset_index(drop=True)
    g = sales_long.groupby("series_idx")["y"]
    for lag in LAGS:
        sales_long[f"lag_{lag}"] = g.shift(lag)
    shifted = g.shift(1)  # never includes the anchor day's own y
    for w in ROLL_WINDOWS:
        roll = shifted.groupby(sales_long["series_idx"]).rolling(w, min_periods=max(2, w // 2))
        sales_long[f"roll_mean_{w}"] = roll.mean().reset_index(level=0, drop=True)
        sales_long[f"roll_std_{w}"] = roll.std().reset_index(level=0, drop=True)
    return sales_long


def encode_categoricals(series_meta: pd.DataFrame) -> pd.DataFrame:
    """Integer-encode static per-series categorical columns. Rebuilt fresh at each origin from
    that origin's own 60-series metadata (categories are fixed store/item/dept/cat/state names,
    identical at every origin, so this needs no persistence across origins)."""
    for c in CAT_COLS:
        cats = sorted(series_meta[c].dropna().unique())
        cat_map = {v: i for i, v in enumerate(cats)}
        series_meta[f"{c}_code"] = series_meta[c].map(cat_map).fillna(-1).astype(int)
    return series_meta


def build_calendar_price_table(cal: pd.DataFrame, sell_prices: pd.DataFrame,
                                series_meta: pd.DataFrame, max_dnum: int) -> pd.DataFrame:
    """Per (series_idx, dnum) table of target-day features, for dnum in 1..max_dnum. Calendar
    and sell_prices both legitimately cover the future horizon in this dataset, so max_dnum may
    extend past the training cutoff without leaking the sales target."""
    cal_feat = build_calendar_features(cal)
    cal_feat = cal_feat[cal_feat["dnum"] <= max_dnum]

    snap_cols = ["snap_MH", "snap_KA", "snap_TN"]
    snap_long = cal_feat[["dnum"] + snap_cols].melt(
        id_vars="dnum", value_vars=snap_cols, var_name="snap_state", value_name="snap"
    )
    snap_long["state_id"] = snap_long["snap_state"].str.replace("snap_", "", regex=False)
    snap_long = snap_long.drop(columns=["snap_state"])

    base = series_meta[["series_idx", "item_id", "store_id", "state_id"]].merge(
        cal_feat[["dnum", "wm_yr_wk", "wday", "month", "days_since_event", "days_until_event"]],
        how="cross",
    )
    base = base.merge(snap_long, on=["dnum", "state_id"], how="left")
    base["snap"] = base["snap"].fillna(0).astype(int)

    base = base.merge(
        sell_prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )
    base = base.sort_values(["series_idx", "dnum"])
    base["sell_price"] = base.groupby("series_idx")["sell_price"].ffill().bfill()

    price_by_wk = (
        base[["series_idx", "wm_yr_wk", "sell_price"]]
        .drop_duplicates(subset=["series_idx", "wm_yr_wk"])
        .sort_values(["series_idx", "wm_yr_wk"])
    )
    price_by_wk["price_prev_wk"] = price_by_wk.groupby("series_idx")["sell_price"].shift(1)
    price_by_wk["price_pct_change"] = (
        price_by_wk["sell_price"] - price_by_wk["price_prev_wk"]
    ) / price_by_wk["price_prev_wk"].replace(0, np.nan)
    base = base.merge(
        price_by_wk[["series_idx", "wm_yr_wk", "price_pct_change"]],
        on=["series_idx", "wm_yr_wk"], how="left",
    )
    base["price_pct_change"] = base["price_pct_change"].fillna(0.0)

    return base[["series_idx", "dnum"] + TARGET_DAY_COLS].reset_index(drop=True)


def prepare_base_frame(sales: pd.DataFrame, cal: pd.DataFrame, sell_prices: pd.DataFrame,
                        n_hist: int, horizon: int = HORIZON):
    """Build:
      - sales_long: (series_idx, dnum, y, lag_*, roll_*) for dnum in 1..n_hist (sales truncated
        to the training cutoff -- this is the only place sales history can leak from, and it is
        capped at n_hist).
      - cal_price: (series_idx, dnum, wday, month, days_since_event, days_until_event, snap,
        sell_price, price_pct_change) for dnum in 1..(n_hist + horizon) -- calendar/price known
        in advance for the whole forecast horizon.
      - series_meta: static per-series categorical codes.
    """
    series_meta = sales[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]].reset_index(drop=True).copy()
    series_meta["series_idx"] = np.arange(len(series_meta))
    series_meta = encode_categoricals(series_meta)

    sales_long = wide_to_long(sales, n_hist)
    sales_long = add_lag_roll_features(sales_long)
    sales_long = sales_long.merge(
        series_meta[["series_idx"] + [f"{c}_code" for c in CAT_COLS]], on="series_idx", how="left"
    )

    cal_price = build_calendar_price_table(cal, sell_prices, series_meta, max_dnum=n_hist + horizon)

    return sales_long, cal_price, series_meta


def make_training_rows_for_bucket(sales_long: pd.DataFrame, cal_price: pd.DataFrame,
                                   cut: int, h_lo: int, h_hi: int) -> pd.DataFrame:
    """For horizon bucket [h_lo, h_hi]: for each h, anchor day = target_day - h. Anchor features
    (lag/roll/cat codes) come from sales_long at the anchor day; target-day features (calendar,
    price) come from cal_price at the target day itself. Only rows with target_day <= cut are
    used for training (both frames are pre-truncated appropriately by the caller anyway, but we
    also filter explicitly here as a second guard against leakage)."""
    anchor_cols = ["series_idx", "dnum"] + ANCHOR_COLS
    rows = []
    for h in range(h_lo, h_hi + 1):
        anchor = sales_long[anchor_cols].copy()
        anchor["target_dnum"] = anchor["dnum"] + h
        anchor["h"] = h
        anchor = anchor.drop(columns=["dnum"])
        rows.append(anchor)
    anchor_feat = pd.concat(rows, ignore_index=True)
    anchor_feat = anchor_feat[anchor_feat["target_dnum"] <= cut]

    merged = anchor_feat.merge(
        cal_price, left_on=["series_idx", "target_dnum"], right_on=["series_idx", "dnum"], how="inner"
    )
    y_map = sales_long[["series_idx", "dnum", "y"]].rename(columns={"dnum": "target_dnum", "y": "y_target"})
    merged = merged.merge(y_map, on=["series_idx", "target_dnum"], how="inner")
    merged = merged.dropna(subset=[f"lag_{MAX_LAG}"])
    return merged


def train_bucket_model(train_feat: pd.DataFrame, val_feat: pd.DataFrame | None = None):
    X = train_feat[FEATURE_COLS]
    y = train_feat["y_target"]
    dtrain = xgb.DMatrix(X, label=y)
    evals = [(dtrain, "train")]
    if val_feat is not None and len(val_feat) > 0:
        dval = xgb.DMatrix(val_feat[FEATURE_COLS], label=val_feat["y_target"])
        evals.append((dval, "val"))
    booster = xgb.train(
        XGB_PARAMS,
        dtrain,
        num_boost_round=N_ESTIMATORS_MAX,
        evals=evals,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS if val_feat is not None and len(val_feat) > 0 else None,
        verbose_eval=False,
    )
    return booster


def forecast_horizon(sales_long: pd.DataFrame, cal_price: pd.DataFrame, cut: int,
                      n_series: int) -> np.ndarray:
    """Train one model per horizon bucket on data up to `cut`, then predict all `HORIZON` days
    for every series. Returns array shaped (n_series, HORIZON) ordered by series_idx."""
    preds = np.zeros((n_series, HORIZON))

    anchor_snap = sales_long[sales_long["dnum"] == cut][["series_idx"] + ANCHOR_COLS].copy()
    anchor_snap = anchor_snap.set_index("series_idx").reindex(range(n_series))

    val_cut = cut - HORIZON

    for h_lo, h_hi in H_BUCKETS:
        train_feat_all = make_training_rows_for_bucket(sales_long, cal_price, cut, h_lo, h_hi)
        if val_cut > MAX_LAG + max(ROLL_WINDOWS):
            train_feat = train_feat_all[train_feat_all["target_dnum"] <= val_cut]
            val_feat = train_feat_all[train_feat_all["target_dnum"] > val_cut]
            if len(train_feat) < 50 or len(val_feat) < 10:
                train_feat, val_feat = train_feat_all, None
        else:
            train_feat, val_feat = train_feat_all, None

        booster = train_bucket_model(train_feat, val_feat)

        for h in range(h_lo, h_hi + 1):
            target_dnum = cut + h
            target_feat = cal_price[cal_price["dnum"] == target_dnum].set_index("series_idx")
            target_feat = target_feat.reindex(range(n_series))

            X_pred = anchor_snap.copy()
            for c in TARGET_DAY_COLS:
                X_pred[c] = target_feat[c].to_numpy()
            X_pred["h"] = h
            X_pred = X_pred[FEATURE_COLS]

            dpred = xgb.DMatrix(X_pred)
            best_it = getattr(booster, "best_iteration", None)
            if best_it is not None:
                p = booster.predict(dpred, iteration_range=(0, best_it + 1))
            else:
                p = booster.predict(dpred)
            preds[:, h - 1] = np.nan_to_num(p, nan=0.0)

    return np.clip(preds, 0, None)
