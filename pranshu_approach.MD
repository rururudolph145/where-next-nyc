# Pranshu's Approach — NYC Yellow Taxi Pickup Density Prediction

Reference notebook: `NYC_Yellow_Taxi_Pickup_Density.ipynb`

## Overview

The author (Pranshu) uses **2015 data as a known baseline** and predicts **2016 pickup counts**
using time-series forecasting methods. The pipeline has 5 stages:

1. Spatial clustering (KMeans)
2. Time binning (10-minute intervals)
3. Smoothing (gap filling)
4. Building ratio/value series
5. Baseline forecasting models (MA, WMA, EMA)

---

## Stage 1: Spatial Clustering

Instead of using the 263 official taxi zones (like our approach), Pranshu clusters raw
lat/lon coordinates into **40 regions** using `MiniBatchKMeans`.

- The number 40 was chosen by finding the smallest cluster count where every pair of
  cluster centroids is at least **1 km apart** (using haversine distance).
- KMeans is fit on 2015 January data, then reused for all subsequent months.
- Each trip gets a `pickup_cluster` label (0–39).

**Our approach:** We use the 263 official `pulocationid` taxi zones, which are more
interpretable and don't require fitting a clustering model.

---

## Stage 2: Time Binning (10-Minute Intervals)

Each pickup's unix timestamp is converted into a **10-minute bin index**:

```python
bin_index = int((pickup_unix - month_start_unix) / 600) + 33
```

- `/ 600` = 600 seconds = 10 minutes per bin
- `+ 33` is an offset (likely aligning to a specific start time)
- January (31 days) = `31 * 24 * 6 = 4464` bins per cluster
- Total data points per month: `40 clusters * 4464 bins = 178,560`

The data is then grouped by `(pickup_cluster, pickup_bin)` and counted.

**Our approach:** We use **1-hour bins** (floor to nearest hour), giving us
`260 zones * hours_in_range` rows. Hourly resolution is more appropriate for
the prediction horizons we target (h+3, h+6, h+12).

---

## Stage 3: Smoothing (Gap Filling)

**Problem:** Not every 10-minute bin has a pickup in every cluster. Many bins are empty
(especially at night or in low-traffic clusters). The groupby only produces entries for
bins that have at least 1 trip, leaving gaps in the time series.

**Solution:** The `smoothing()` function fills these gaps by **linearly redistributing**
counts from neighboring non-empty bins:

- For a gap between two known bins, it takes the sum of the left and right neighbor counts,
  divides evenly across the gap + the two neighbors, and assigns `ceil()` of that value.
- For a trailing gap (no right neighbor), it spreads the last known count across the
  remaining bins.
- For a leading gap (no left neighbor), it spreads the first known count backwards.

This is essentially a crude **linear interpolation** that ensures every cluster has
exactly 4464 values (one per 10-minute slot) with no gaps.

**Our approach:** We build a complete grid of `(zone, hour)` via `MultiIndex.from_product`
and fill missing combinations with 0. We don't interpolate — a zero means no pickups
happened, which is a valid signal for demand forecasting.

---

## Stage 4: Building the Ratio Series

After smoothing, the author constructs a DataFrame with three columns:

| Column       | Meaning                                    |
|--------------|--------------------------------------------|
| `Given`      | Smoothed 2015 pickup count for bin t       |
| `Prediction` | Smoothed 2016 pickup count for bin t       |
| `Ratios`     | `Prediction / Given` = year-over-year ratio |

```python
ratios['Given']      = jan_2015_smooth      # 40*4464 = 178,560 values
ratios['Prediction'] = jan_2016_smooth
ratios['Ratios']     = Prediction / Given
```

The idea: **the ratio between 2016 and 2015 for the same time slot is more stable
and predictable than the raw count.** If you can predict tomorrow's ratio, you multiply
it by the known 2015 baseline to get the 2016 forecast.

---

## Stage 5: Baseline Forecasting Models

All models iterate over the flat array of `40 clusters * 4464 bins = 178,560` values.
Each cluster's bins are processed sequentially (the loop resets at every `i % 4464 == 0`).

### 5a. Simple Moving Average (MA)

Equal weight to the last N values.

**MA on Ratios (MA_R)** — window = 3 (= 30 minutes lookback):

```
R_t = (R_{t-1} + R_{t-2} + R_{t-3}) / 3
Predicted_2016_t = Given_2015_t * R_t
```

**MA on Values (MA_P)** — window = 1 (= 10 minutes lookback):

```
P_t = P_{t-1}
```

This is just a naive "last value" predictor.

### 5b. Weighted Moving Average (WMA)

Recent values get higher weights (linearly decreasing).

**WMA on Ratios (WA_R)** — window = 5 (= 50 minutes lookback):

```
R_t = (5*R_{t-1} + 4*R_{t-2} + 3*R_{t-3} + 2*R_{t-4} + 1*R_{t-5}) / 15
Predicted_2016_t = Given_2015_t * R_t
```

**WMA on Values (WA_P)** — window = 2 (= 20 minutes lookback):

```
P_t = (2*P_{t-1} + 1*P_{t-2}) / 3
```

### 5c. Exponential Moving Average (EMA)

Single hyperparameter `alpha` controls both the effective window size and weight decay.
The effective lookback is approximately `1 / (1 - alpha)` bins.

**EMA Variant 1 on Ratios (EA_R1)** — alpha = 0.6 (~25 min effective lookback):

```
R'_t = alpha * R'_{t-1} + (1 - alpha) * R_{t-1}
     = 0.6 * R'_{t-1} + 0.4 * R_{t-1}
```

**EMA Variant 1 on Values (EA_P1)** — alpha = 0.3 (~14 min effective lookback):

```
P'_t = alpha * P'_{t-1} + (1 - alpha) * P_{t-1}
     = 0.3 * P'_{t-1} + 0.7 * P_{t-1}
```

**EMA Variant 2 on Ratios (EA_R2)** — alpha = 0.5 (~20 min effective lookback):

```
R'_t = R'_{t-1} + alpha * (R_{t-1} - R'_{t-1})
```

(Mathematically equivalent to variant 1 but written differently.)

**EMA Variant 2 on Values (EA_P2)** — alpha = 0.5 (~20 min effective lookback):

```
P'_t = P'_{t-1} + alpha * (P_{t-1} - P'_{t-1})
```

### 5d. Corrected EMA (EA_R2_Corrected)

Same as EA_R2, but with a **spike correction heuristic**:

- If `|predicted - previous_actual| > 120`: fall back to `Given_t - Given_{t-1}`
- If that's still off by more than 50: fall back to `previous_actual`

This caps wild swings caused by ratio instability in low-traffic bins.

---

## Model Comparison

The author uses **MAPE** (Mean Absolute Percentage Error) and **MSE** to compare models
on January 2016 data.

**Winner:** EMA Variant 1 on raw 2016 values (EA_P1):

```
P'_t = 0.3 * P'_{t-1} + 0.7 * P_{t-1}
```

This makes intuitive sense — at 10-minute granularity, the most recent actual value is
the strongest predictor of the next bin.

---

## Summary of All Lookback Windows

| Model    | Variant | Window/Alpha | Lookback (time)  |
|----------|---------|-------------|------------------|
| MA_R     | Ratio   | n=3         | 30 min           |
| MA_P     | Value   | n=1         | 10 min           |
| WA_R     | Ratio   | n=5         | 50 min           |
| WA_P     | Value   | n=2         | 20 min           |
| EA_R1    | Ratio   | alpha=0.6   | ~25 min          |
| EA_P1    | Value   | alpha=0.3   | ~14 min          |
| EA_R2    | Ratio   | alpha=0.5   | ~20 min          |
| EA_P2    | Value   | alpha=0.5   | ~20 min          |

All hyperparameters were **manually tuned**.

---

## Key Differences from Our Approach

| Aspect            | Pranshu                             | Ours                                  |
|-------------------|-------------------------------------|---------------------------------------|
| Spatial unit      | 40 KMeans clusters                  | 260 official taxi zones               |
| Time granularity  | 10-minute bins                      | 1-hour bins                           |
| Gap handling      | Linear interpolation (smoothing)    | Fill with 0 (zero = no demand)        |
| Prediction target | Year-over-year ratio or raw count   | Raw pickup count (trip_count)         |
| Baseline required | Yes (needs same-period prior year)  | No (uses own lag features)            |
| Forecast horizon  | Next 10-min bin                     | h+3, h+6, h+12 hours                 |
| Feature approach  | Handcrafted MA/WMA/EMA              | Lag features + ML model (LightGBM)    |
