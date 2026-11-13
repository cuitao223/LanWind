# Local data contract / 本机数据接口

No datasets or example arrays are distributed here. Prepare local `.npz` bundles from your own authorized observations and already-issued forecasts. Never commit those files.

本仓库不提供数据集、观测样本或回放数组。所有数据由使用者在本机独立准备，保存在仓库外或已忽略的目录。

## Input schema

| Key | Shape | Meaning |
|---|---|---|
| `history_uv_ms` | `[T,2,5,33,33]`, float32 | Past u/v in m/s; NaN for missing |
| `history_times_utc` | `[T]`, int64 | Explicit Unix UTC seconds, strictly increasing |
| `gfs_uv_ms` | `[60,2,33,33]`, float32 | Already-issued GFS background on the decoder clock |
| `gfs_lead_seconds` | `[60]` | Historical median cadence times step indices 1..60 |
| `gfs_available_at_utc` | scalar int64 | Actual availability, no later than the origin |
| `future_uv_ms` | `[L,2,5,33,33]` | Actual targets; optional for inference, NaN for missing |
| `future_times_utc` | `[L]`, int64 | Actual target timestamps; required with targets |

`T` is 2–23. Native history brackets the two-hour boundary: at most one older scan is retained, by at most 900 s. Historical gaps cannot exceed 900 s. Targets use 1–60 actual observations through the first scan at or after six hours, strictly after the origin, without gaps over 900 s. Future observations are never used for feature generation. All five structural height slots must be present; unavailable heights remain NaN.

Channel order is **u×5, v×5, mask×5, Δt×5**. Spatial order is **z,y,x**. Radar-centered x points east, y north; units metres. Cell centers span −3200..3200 m at 200 m spacing. Heights 100/150/200/250/300 m are radar-relative. u is eastward, v northward. Meteorological from-direction is `atan2(-u,-v)` modulo 360 degrees. Vertical wind w is excluded.

## Missingness and normalization

Both wind components must be finite for a pair to be valid. Missing normalized values become zero with an explicit zero observation mask. Δt is the per-cell interval since the previous valid observation inside the window, divided by its training mean. A missing predecessor is −1, distinct from every positive interval.

Residual baseline uses the latest valid per-cell u/v. Evaluation requires that baseline to exist. Missing targets do not contribute to loss or metrics. The released config stores original training-only normalization. New training refits on its own training partition and saves separate settings.

## GFS and time alignment

Use NOAA GFS 0.25° operational u/v at 100 m above ground. Reference policy: previous 6-hourly cycle at least 6 hours old, with a 15-minute storage availability margin. Retain source/cycle records locally. `gfs_available_at_utc` is a caller-provided provenance assertion: software checks its value but cannot establish the origin of arbitrary arrays.

Spatially map known forecast fields to the radar grid and interpolate them onto the nominal decoder clock. Do not interpolate radar target labels, use future analyses, or infer timezones from wind strength. The decoder clock comes from historical cadence. Sixty steps cover approximately six hours under the original sampling cadence, not a forced fixed 6-minute interval.

## Chronological partitions

Allocate complete time ranges 70%/15%/15%. Each history-through-target window stays inside one range and avoids long gaps. Local layout:

```text
/private/bundles/
  train/
  val/
  test/
```

Training rejects overlap between complete training and validation spans and never opens `test/`. Evaluate only after selecting weights. The caller must ensure test windows remain in a held-out range.

## Sources

Historical radar: Jiaozhou Airport CAPPI horizontal wind products, obtained from the data holder. No verified public download address is available. Request authorization from the holder.

Background: https://registry.opendata.aws/noaa-gfs-bdp-pds/ .

Do not commit NetCDF, GRIB, NPY, NPZ, HDF5, archives, per-frame inventories or prediction bundles. Release checks reject data containers, including force-added ignored files.
