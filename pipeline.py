#!/usr/bin/env python3
"""
pipeline.py — RetailCast India forecasting pipeline.

Approach (full reasoning and evidence in approach_summary.md / chat_export.md):

- market_signal.csv is EXCLUDED entirely. It only covers d_1..d_1913 (zero rows in the horizon,
  so it is not available at prediction time), and per-series it is a near-deterministic linear
  rescaling of same-day sales: per-id correlation with actual sales is 0.87-0.96, the
  signal/sales ratio clusters tightly per series (~9.9-11.0x), and mkt_signal is exactly 0 on all
  45,970 rows where sales==0. That is the target leaking back to us as a "feature" — not a
  genuine independent market signal.

- vendor_signal.csv DOES cover the full horizon (d_1..d_1941) but historically tracks actual
  sales poorly (mean per-id correlation ~0.12 in-history, ~50.5% median MAPE (~75% mean) over the trailing 90 days,
  and it does not react to the same regime shifts visible in sales_train). It is available at
  prediction time but not trustworthy enough to use as a model input; it was only used as an
  independent sanity check.

- Level and weekday seasonality are estimated from a trailing window (56 days), not full
  1,913-day history, because several series show a genuine regime change: HOMECARE_2_AGARBATTI
  ramps to a materially higher level in the last ~90 days at every store (root-caused: the item
  is priced and lightly transacting from d_1, price stays flat through the ramp, and the
  changepoint is staggered per store rather than a single chain-wide day — a gradual
  distribution/velocity increase of an existing SKU, not a new listing), and some individual
  store series decay toward zero in the most recent quarter (e.g. ELECTRONICS_1_CABLE_TN_2,
  GROCERY_3_PICKLE_KA_1/2/3 — price is flat or even falls through the decline, and non-zero
  sales still occur in the final weeks, ruling out delisting or a terminal stockout; this reads
  as organic, region-specific demand softening, not a supply-side artifact). Whatever the
  mechanism, a full-history average would badly misforecast both directions; a trailing window
  adapts to whatever level each series is actually at now without needing to resolve the cause.

- The trailing-window seasonal estimate is blended with a simple flat trailing-14-day mean
  (40/60, weighted toward the flat mean) because a pure weekday-seasonal fit overfits noisy
  per-weekday counts for low-volume/intermittent series; this blend and window were chosen by
  grid search on mean RMSSE across nine backtest origins (28 through 252 days back), not a
  single origin.

- Calendar events do NOT lift all three states equally, so a single pooled event-type
  multiplier is wrong. Pan-India National events (Diwali, Independence Day, Gandhi Jayanti)
  and most Religious/Cultural events show a large, consistent lift across MH/KA/TN and are fine
  pooled. But the two events that actually fall in the forecast horizon diverge by state:
  Ram_Navami (d_1921) is flat everywhere historically (MH +3%, KA -2%, TN +2%) — a pooled
  1.25x used previously badly overstated it — while Eid_al_Fitr (d_1928) shows real
  state-specific lift (MH +49%, KA +6%, TN +22%). Both are now applied as per-(event, state)
  multipliers rather than one flat per-event-type number. Sporting events (IPL_Final) actually
  show a consistent *negative* lift (-16% to -32% across states) historically, but no Sporting
  event falls in this horizon so it has no effect on the submission. SNAP flags showed ~no
  effect (<1% mean difference) and are not used.

- Hybrid model split (see backtest.py / approach_summary.md Q4-Q5 for the backtest): the
  trailing-window per-series baseline above is used ONLY for the 15 series flagged with a proven
  regime change
  (HOMECARE_2_AGARBATTI x10 stores, ELECTRONICS_1_CABLE_KA_3/TN_2, GROCERY_3_PICKLE_KA_1/2/3),
  where pooling across series risks re-learning the pre-change level as signal. For the other 45
  series with no such evidence, a pooled XGBoost/Tweedie direct-multistep model
  (gbm_model.py) is used instead — backtesting confirms it's the
  stronger model specifically on the stable subset (mean per-series RMSSE 0.746 vs. 0.753 for
  the baseline), while the baseline remains stronger on the regime subset (1.313 vs. 1.327).
  Blending per-segment beats either pure approach: hybrid mean RMSSE 0.888 / WAPE 0.449 vs.
  baseline-only 0.893 / 0.466 and GBM-only 0.891 / 0.450 across the same 9-origin backtest.

Usage:
    python3 pipeline.py --data data --out submission.csv

Run from the repo root (retailcast_submission/); data/ is a subfolder of this repo.
"""
import argparse
import os

import numpy as np
import pandas as pd

from gbm_model import prepare_base_frame, forecast_horizon as gbm_forecast_horizon

# Series with documented evidence of a regime change (see approach_summary.md Q2) -- these keep
# the trailing-window per-series baseline; every other series uses the pooled GBM instead.
REGIME_SERIES_IDS = {
    "HOMECARE_2_AGARBATTI_MH_1_validation", "HOMECARE_2_AGARBATTI_MH_2_validation",
    "HOMECARE_2_AGARBATTI_MH_3_validation", "HOMECARE_2_AGARBATTI_MH_4_validation",
    "HOMECARE_2_AGARBATTI_KA_1_validation", "HOMECARE_2_AGARBATTI_KA_2_validation",
    "HOMECARE_2_AGARBATTI_KA_3_validation", "HOMECARE_2_AGARBATTI_TN_1_validation",
    "HOMECARE_2_AGARBATTI_TN_2_validation", "HOMECARE_2_AGARBATTI_TN_3_validation",
    "ELECTRONICS_1_CABLE_KA_3_validation", "ELECTRONICS_1_CABLE_TN_2_validation",
    "GROCERY_3_PICKLE_KA_1_validation", "GROCERY_3_PICKLE_KA_2_validation",
    "GROCERY_3_PICKLE_KA_3_validation",
}

N_HIST = 1913
HORIZON = 28
FCOLS = [f"F{i}" for i in range(1, HORIZON + 1)]

SEASONAL_WINDOW = 56   # trailing days used for level + weekday-seasonal fit
FLAT_WINDOW = 14       # trailing days used for the flat-mean component
BLEND_ALPHA = 0.4      # weight on the seasonal component (1 - alpha on the flat component)
WDAY_SHRINK_N = 5       # shrinkage strength (pseudo-observations) for weekday ratios toward 1.0
# Per-event, per-state multipliers, derived from weekday-adjusted historical lift
# (event-day mean sales vs. same-state non-event baseline, matched by weekday). Pan-India
# National events (Diwali, Independence Day, Gandhi Jayanti, Dussehra, Ganesh Chaturthi) show
# a large, consistent lift across all three states, so a single pooled multiplier is fine for
# those. But the two events that actually fall in this horizon do NOT behave uniformly:
#   Ram_Navami (d_1921): flat everywhere historically (MH +3%, KA -2%, TN +2%) -- a pooled
#     1.25x for "Religious" would badly overstate it.
#   Eid_al_Fitr (d_1928): large state divergence (MH +49%, KA +6%, TN +22%) -- a single pooled
#     multiplier either understates MH or overstates KA.
# so these two are keyed by (event_name, state) rather than by event_type.
EVENT_STATE_LIFT = {
    ("Ram_Navami", "MH"): 1.03,
    ("Ram_Navami", "KA"): 0.98,
    ("Ram_Navami", "TN"): 1.02,
    ("Eid_al_Fitr", "MH"): 1.45,
    ("Eid_al_Fitr", "KA"): 1.06,
    ("Eid_al_Fitr", "TN"): 1.20,
}
# Fallback for any other National/Religious/Cultural event that might appear in the horizon in
# future runs (this dataset's own horizon only contains the two events above). Kept deliberately
# modest -- pooled type-level lift measured on history overstates single named events like
# Ram_Navami, so this is a conservative floor, not the 1.25x originally used.
EVENT_LIFT_FALLBACK = {"National": 1.15, "Religious": 1.10, "Cultural": 1.08}
EVENT_LIFT_CAP = 1.5


def load_calendar(data_dir: str) -> pd.DataFrame:
    cal = pd.read_csv(os.path.join(data_dir, "calendar.csv"))
    cal["dnum"] = cal["d"].str.replace("d_", "", regex=False).astype(int)
    return cal


def weekday_seasonal_forecast(Ytr: np.ndarray, window: int, wday_hist: np.ndarray,
                               horizon_wday: np.ndarray, wshrink_n: float) -> np.ndarray:
    """Trailing-window level x shrunk weekday-seasonal index."""
    n = Ytr.shape[0]
    recent = Ytr[:, -window:]
    wday_recent = wday_hist[-window:]
    series_mean = recent.mean(axis=1)

    wday_idx = np.ones((n, 8))  # index 1..7 used (wday convention: 1=Sat..7=Fri)
    for wd in range(1, 8):
        cols = wday_recent == wd
        if cols.sum() == 0:
            continue
        wmean = recent[:, cols].mean(axis=1)
        ratio = np.divide(wmean, series_mean, out=np.ones_like(wmean), where=series_mean > 0)
        n_obs = cols.sum()
        w_shrink = n_obs / (n_obs + wshrink_n)
        wday_idx[:, wd] = w_shrink * ratio + (1 - w_shrink) * 1.0
    wday_idx[:, 1:] = wday_idx[:, 1:] / wday_idx[:, 1:].mean(axis=1, keepdims=True)

    P = np.zeros((n, len(horizon_wday)))
    for j, wd in enumerate(horizon_wday):
        P[:, j] = series_mean * wday_idx[:, wd]
    return P


def baseline_forecast(sales: pd.DataFrame, cal: pd.DataFrame) -> np.ndarray:
    """Trailing-window per-series statistical model (see module docstring). Used for the
    regime-flagged series; also the fallback if the pooled GBM cannot be trained."""
    dcols = [f"d_{i + 1}" for i in range(N_HIST)]
    Y = sales[dcols].to_numpy(float)
    states = sales["state_id"].to_numpy()

    hist_cal = cal[cal["dnum"] <= N_HIST].sort_values("dnum")
    wday_hist = hist_cal["wday"].to_numpy()

    horizon_cal = cal[(cal["dnum"] > N_HIST) & (cal["dnum"] <= N_HIST + HORIZON)].sort_values("dnum")
    horizon_wday = horizon_cal["wday"].to_numpy()
    horizon_event_name = horizon_cal["event_name_1"].fillna("none").to_numpy()
    horizon_event_type = horizon_cal["event_type_1"].fillna("none").to_numpy()

    P_seasonal = weekday_seasonal_forecast(Y, SEASONAL_WINDOW, wday_hist, horizon_wday, WDAY_SHRINK_N)
    flat = Y[:, -FLAT_WINDOW:].mean(axis=1, keepdims=True)
    P_flat = np.tile(flat, (1, HORIZON))

    P = BLEND_ALPHA * P_seasonal + (1 - BLEND_ALPHA) * P_flat

    for j, (ename, etype) in enumerate(zip(horizon_event_name, horizon_event_type)):
        if ename == "none":
            continue
        for i, st in enumerate(states):
            mult = EVENT_STATE_LIFT.get((ename, st))
            if mult is None:
                mult = EVENT_LIFT_FALLBACK.get(etype)
            if mult is not None:
                P[i, j] *= min(mult, EVENT_LIFT_CAP)

    return np.clip(P, 0, None)


def gbm_forecast(sales: pd.DataFrame, cal: pd.DataFrame, sell_prices: pd.DataFrame) -> np.ndarray:
    """Pooled XGBoost/Tweedie direct-multistep model (gbm_model.py). Used for the 45 series
    with no evidence of a regime change, where cross-series pooling is safe."""
    sales_long, cal_price, _ = prepare_base_frame(sales, cal, sell_prices, n_hist=N_HIST, horizon=HORIZON)
    return gbm_forecast_horizon(sales_long, cal_price, N_HIST, n_series=len(sales))


def compute_forecast(sales: pd.DataFrame, cal: pd.DataFrame, sell_prices: pd.DataFrame) -> np.ndarray:
    """Hybrid: baseline for regime-flagged series, pooled GBM for the rest (see module
    docstring and approach_summary.md Q4-Q5 for the backtest that justified this split)."""
    P_base = baseline_forecast(sales, cal)
    P_gbm = gbm_forecast(sales, cal, sell_prices)
    is_regime = sales["id"].isin(REGIME_SERIES_IDS).to_numpy()
    return np.where(is_regime[:, None], P_base, P_gbm)


def main(data_dir: str, out_path: str) -> None:
    sales_path = os.path.join(data_dir, "sales_train.csv")
    sales = pd.read_csv(sales_path)
    cal = load_calendar(data_dir)
    sell_prices = pd.read_csv(os.path.join(data_dir, "sell_prices.csv"))
    # market_signal.csv is intentionally never read (target leakage, no horizon coverage).
    # vendor_signal.csv is intentionally never read (weak historical fit; see docstring above).
    # sell_prices.csv is used ONLY as a target-day feature (weekly price + pct-change) for the
    # pooled GBM on the non-regime series -- both are legitimately known in advance for the
    # forecast horizon in this dataset (see approach_summary.md).

    forecast = compute_forecast(sales, cal, sell_prices)

    submission = pd.DataFrame(forecast, columns=FCOLS)
    submission.insert(0, "id", sales["id"].values)
    submission.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(submission)} rows)")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(here, "data"))
    ap.add_argument("--out", default=os.path.join(here, "submission.csv"))
    args = ap.parse_args()
    main(args.data, args.out)
