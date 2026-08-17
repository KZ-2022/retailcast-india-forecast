# RetailCast India — Data Dictionary

Reference documentation for the dataset files provided in this challenge. All files describe a
realistic Indian retail panel: **6 products × 10 stores = 60 series**, with **1,913 days** of daily
sales history (`d_1` … `d_1913`). You forecast the next **28 days** (`d_1914` … `d_1941`).

Stores span three states: Maharashtra (`MH_1`–`MH_4`), Karnataka (`KA_1`–`KA_3`), Tamil Nadu
(`TN_1`–`TN_3`). Products: `ELECTRONICS_1_CABLE`, `ELECTRONICS_1_CHARGER`, `GROCERY_3_ATTA`,
`GROCERY_3_PICKLE`, `HOMECARE_1_DETERGENT`, `HOMECARE_2_AGARBATTI`.

---

## sales_train.csv
Daily unit sales per series. One row per series (60 rows).

| Field | Type | Description |
|---|---|---|
| `id` | string | Series id: `{item_id}_{store_id}_validation` |
| `item_id` | string | Product id (e.g. `GROCERY_3_PICKLE`) |
| `dept_id` | string | Department (e.g. `GROCERY_3`) |
| `cat_id` | string | Category (e.g. `GROCERY`) |
| `store_id` | string | Store id (e.g. `MH_2`) |
| `state_id` | string | State (`MH`, `KA`, `TN`) |
| `d_1` … `d_1913` | integer | Units sold on each day (day index) |

## calendar.csv
Maps day index to date and calendar features. Covers `d_1`…`d_1969` (history + horizon + buffer).

| Field | Type | Description |
|---|---|---|
| `date` | string | Calendar date (YYYY-MM-DD) |
| `wm_yr_wk` | integer | Retail week id (used to join `sell_prices.csv`) |
| `weekday` | string | Day name |
| `wday` | integer | Day-of-week index (1 = Saturday … 7 = Friday) |
| `month`, `year` | integer | Month, year |
| `d` | string | Day index (`d_N`) |
| `event_name_1`, `event_type_1` | string | Primary festival/holiday and its type (may be empty) |
| `event_name_2`, `event_type_2` | string | Secondary event (rare; may be empty) |
| `snap_MH`, `snap_KA`, `snap_TN` | integer | Assistance-program purchase-eligibility flag per state (0/1) |

## sell_prices.csv
Weekly price per item per store. Join to sales via (`store_id`, `item_id`, `wm_yr_wk`).

| Field | Type | Description |
|---|---|---|
| `store_id` | string | Store id |
| `item_id` | string | Product id |
| `wm_yr_wk` | integer | Retail week id |
| `sell_price` | float | Unit price that week (absent rows = not sold that week) |

## market_signal.csv
A supplied market-demand index per series per day. Long format.

| Field | Type | Description |
|---|---|---|
| `id` | string | Series id |
| `d` | string | Day index (`d_N`) |
| `mkt_signal` | float | A market signal value for that series/day |

> Provided as a candidate feature. As with any feature, consider what it represents, how it was
> produced, and whether it will be available at prediction time before you rely on it.

## sample_submission.csv
The exact required output format. **60 rows**, columns `id`, `F1`…`F28` (day 1…28 of the horizon).
Replace the zeros with your forecasts (non-negative; fractional allowed). Ids must match verbatim;
row order does not matter (it is aligned by `id`).

### Vendor feeds — validate before use
Two third-party vendor feeds are included. They are NOT interchangeable; check each one's coverage, timing, and provenance before deciding whether to use it.

- **`market_signal.csv`** (`id`, `d`, `mkt_signal`): a demand index. Inspect its coverage and how it relates in time to sales before relying on it.
- **`vendor_signal.csv`** (`id`, `d`, `vendor_forecast`): a vendor-supplied baseline demand forecast. Inspect its coverage and how it relates in time to sales before relying on it.

> Not every feed you are handed is safe to use as a feature. Some may not be available for the forecast horizon, or may be derived from the very thing you are predicting. That is your call to make.
