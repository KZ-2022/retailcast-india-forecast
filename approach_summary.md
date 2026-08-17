# RetailCast India — Approach Summary / Technical Decision Log

## Q1. Audit method (~150 words)

I audited in this order: (1) shape/coverage checks — row counts, day-index ranges, per-id row
counts for `market_signal.csv`/`vendor_signal.csv` — to see which feeds reach the forecast
horizon (`d_1914`–`d_1941`). (2) Per-series profiling of `sales_train.csv`: full-history vs.
trailing-90/120-day mean ratio, and max zero-run length, to find regime changes and stockouts.
(3) Feed provenance: same-day correlation and ratio of `mkt_signal`/`vendor_forecast` against
sales, per id and pooled, plus `mkt_signal` on exactly-zero-sales rows. (4) Calendar checks: mean
sales on SNAP vs. non-SNAP days, by `event_type_1`, then per state for asymmetry. (5) Price
checks: `sell_price` volatility, horizon-week coverage, and price behavior around each regime
changepoint. Done once every feed's coverage was verified and every anomaly had a quantified
verdict.

## Q2. Data verdicts (~500 words)

**market_signal.csv — target leakage, excluded.** What: all 60 series, `d_1`–`d_1913` only.
Evidence: `mkt_signal` has zero rows for `d_1914`+, so it can't be a prediction-time feature
regardless of quality. Independently, per-series correlation with same-day sales is 0.87–0.96
(pooled 0.94), the ratio `mkt_signal/sales` clusters per series near-constant (per-id mean
≈9.9–11.0), and on all 45,970 rows where `sales == 0`, `mkt_signal` is exactly `0.0`, no
exceptions. Action: excluded entirely, not even as a training-time regressor — a feature this
derived from the target would bias model selection even if dropped later. Rejected reading: "a
genuine demand index that happens to correlate with sales" — a real index wouldn't be exactly
zero on every one of 46k zero-sales rows; that's a deterministic target transform.

**vendor_signal.csv — weak feed, excluded as a model input.** What: all 60 series, full horizon
coverage (`d_1`–`d_1941`). Evidence: mean per-id correlation with actual sales in-history is only
~0.12 (vs. 0.94 for `mkt_signal`), MAPE over the trailing 90 days is ~50.5% median (~75% mean),
and on AGARBATTI, horizon-period vendor forecasts sit below the series' own recent 90-day mean —
the vendor model under-adapted too. Action: excluded, kept only as an informal cross-check.
Rejected reading: "free ensemble diversity" — its error correlates with the same regime-shift
problem already corrected for; blending it in risked reintroducing staleness.

**Regime changes in `sales_train.csv` — 15 series flagged, given a different model (Q4).** What:
`HOMECARE_2_AGARBATTI` (all 10 stores) sits at ~0 units for ~1,300 days then ramps to a
materially higher level over the remaining history (trailing90/full-history ratio 1.9x–6.6x
across stores, staggered changepoints roughly day 1293–1317 of 1913). Conversely
`ELECTRONICS_1_CABLE_KA_3`/`TN_2` (ratio 0.56/0.29) and three `GROCERY_3_PICKLE` Karnataka stores
(ratio 0.46/0.39/0.22) decline sharply in the final quarter; most other CABLE/PICKLE stores held
steady or grew. Mechanism, checked in `sell_prices.csv`: AGARBATTI is priced and lightly
transacting from `d_1` in every store, price stays flat through the ramp, and the changepoint is
staggered per store (not one chain-wide day) — a gradual velocity increase, not a launch. For
CABLE/PICKLE, price shows no correlation with severity (`PICKLE_KA_3`, worst decline, flat price;
`PICKLE_KA_1`, mildest, largest price rise, +17%) and non-zero sales persist in the final weeks,
ruling out delisting or stockout. Action: these 15 keep a trailing-window model (Q4) so
level/seasonality recompute from each series' own recent window; the other 45 are safe to pool.
Rejected reading: "AGARBATTI is a new listing, CABLE/PICKLE-KA is a price-driven collapse" —
price/early sales predate the ramp by years, and price moves don't track severity.
**Calendar events do not lift all three states equally — pooled multiplier replaced with
per-(event, state).** What: the two events in the forecast horizon, Ram Navami (`d_1921`) and
Eid al-Fitr (`d_1928`), both tagged `Religious`. Evidence: weekday-matched per-state lift shows
Ram Navami flat everywhere (MH +3%, KA −2%, TN +2%) while Eid al-Fitr diverges sharply (MH +49%,
KA +6%, TN +22%) — a pooled 1.25× overstates Ram Navami and is wrong in different directions for
Eid al-Fitr per state. Pan-India events (Diwali, Independence Day) stay tighter across states, so
pooling is fine there. Action: per-(event_name, state) multipliers: 1.03/0.98/1.02 (MH/KA/TN) for
Ram Navami, 1.45/1.06/1.20 for Eid al-Fitr; a smaller type-level fallback (1.15/1.10/1.08) covers
any other future-horizon event. Rejected reading: "pan-India religious holidays warrant one
national multiplier" — Eid al-Fitr's lift tracks state Muslim population share, not a uniform
effect. SNAP flags showed <1% mean difference and were not used.

## Q3. What I left alone (~150 words)

The `GROCERY_3_PICKLE_MH_2` price history has two isolated one-week price collapses (₹4.34 →
₹1.20 → ₹4.34) at `wm_yr_wk` 2040 and again at 2315 — a 72% swing each time, looking like a
data-entry error or an unlabeled promo. I left it uncorrected. Sell price never entered the model
for this series (only 4 unique price-weeks in the horizon, no comparable promo pattern in history
to calibrate elasticity), so this anomaly has zero effect on the submitted forecast either way.
"Fixing" it would mean guessing at a true price with no ground truth — the kind of
confident-but-unjustified correction the brief warns against. Restraint was correct: the anomaly
is isolated to a feature I don't use, not something contaminating the target.

## Q4. Modelling choices (~250 words)

The final model is a hybrid, split by the Q2 regime findings. For the 15 series with a proven
regime change (`HOMECARE_2_AGARBATTI` x10, `ELECTRONICS_1_CABLE_KA_3`/`TN_2`,
`GROCERY_3_PICKLE_KA_1/2/3`) I kept a per-series statistical baseline: trailing-56-day
weekday-seasonal index (shrunk toward 1.0, pseudo-count 5) times trailing-56-day mean level,
blended 40/60 with a flat trailing-14-day mean, then a capped event multiplier — level recomputes
from each series' own window, so no cross-series signal contaminates a just-launched or
discontinued series. For the other 45 I tested rather than assumed the pooled-model rejection: a
pooled XGBoost/Tweedie direct-multistep model, backtested identically (`backtest.py`). Segmenting
error by regime status shows the baseline wins on the 15 regime series (1.313 vs. GBM's 1.327)
while GBM wins on the other 45 (0.746 vs. 0.753) — neither wins outright pooled (baseline 0.893
mean RMSSE vs. GBM 0.891), but routing each segment to its stronger model beats both: hybrid mean
RMSSE 0.888. `market_signal`/`vendor_signal` excluded per Q2; prices/SNAP excluded per Q2/Q3
except as a target-day feature inside the GBM segment, which also sees calendar directly. The
baseline segment applies calendar events as per-(event, state) multipliers instead.

## Q5. Validation you trust (~200 words)

I backtested by truncating history at nine origins (28/56/84/112/140/168/196/224/252 days before
`d_1913`) and forecasting 28 days ahead from each, scoring with the same RMSSE (scaled by
in-sample naive first-difference RMSE) and WAPE the challenge uses (reproducible via
`python3 backtest.py --data data`). I grid-searched the baseline's hyperparameters against the
mean RMSSE across all nine origins, then re-verified the hybrid split the same way once the GBM
candidate existed — never on a single origin. Per-origin hybrid RMSSE: 0.817, 0.818, 0.875, 0.877,
0.927, 0.919, 0.974, 0.882, 0.901 (mean 0.888); naive seasonal RMSSE at the same origins is always
higher (1.09–1.36). Mean WAPE is ≈0.449. The hybrid beats seasonal-naive by a wide, stable margin
at every origin, not just the most recent, and beats both single-model alternatives it was built
from. I expect roughly RMSSE ≈ 0.85–1.0 and WAPE ≈ 0.40–0.55 on the true held-out horizon. What
could make local validation look better than reality: testing only the most recent window, or
tuning against the same window used to report the final number. I protected against both by
using nine independent origins spanning most of a year and selecting on the mean across all nine.

## Q6. Your least-sure call (~150 words)

The 15-series regime cutoff is the call I'd revisit first: a hard boundary from a
full-history-vs-trailing-90-day ratio threshold (>2x or <0.5x); a series just under it gets the
GBM even though its own recent behavior might make the baseline safer, and vice versa. I checked
whether the two harder-than-neighbors origins (140/196 days back, RMSSE 0.927/0.974) were driven
by the regime segment specifically — they're not; both segments worsen together there, suggesting
a harder demand period, not a bad split. Given more time I'd make the cutoff a continuous
confidence weight blending baseline and GBM per series rather than a binary switch. Meanwhile I
hedged by picking the threshold from the same regime evidence already root-caused in Q2, not a
threshold tuned to this backtest's outcome.

A related limitation noticed post-hoc: the pooled GBM smooths the ~21 zero-inflated
`ELECTRONICS_1_CABLE_*` series (>50% historical zero-days) into a near-constant small daily
fraction rather than a realistic zero/spike pattern — Tweedie rewards this on RMSSE/WAPE even
though it isn't a faithful demand shape, wider than the 15 series I flagged. Ratio evidence
didn't support calling it a regime change, so I left it; a dedicated intermittent-demand method
(Croston/TSB) for this cluster is the next thing I'd try.

## Q7. Reproduce and stress (~100 words)

`python3 pipeline.py --data data --out submission.csv` regenerates the file
(`python3 backtest.py --data data` reproduces the Q5 table). If a new series next month showed
the same discontinuation pattern as `ELECTRONICS_1_CABLE_TN_2`, it would still get the GBM by
default (not in `REGIME_SERIES_IDS`) and be wrong until someone re-ran the regime check. It also
wouldn't catch a new leakage feed with a different signature than `mkt_signal`'s exact-zero
pattern, or a new event's per-state divergence — all three still need a human look.
