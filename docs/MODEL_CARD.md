# LanWind model card

## Model

Radar/GFS-conditioned residual ConvLSTM for multilevel horizontal wind forecasting. Code and a tensor-only state dictionary are released; datasets, input examples, decoded caches and replay bundles are excluded.

Two encoder cells use 64 and 32 channels. Two distinct decoder cells also use 64 and 32 channels. Kernels are 3×3. A 1×1 head produces 10 delta channels. Parameter count: 639,082. Five structural heights are jointly represented.

The previous two hours contain at most 23 native scans; the decoder emits 60 nominal steps. Encoder h/c initialize corresponding decoder layers. Fixed context plus lead embedding, linear time projection and GFS projection drive each step. Hidden states recur; predicted winds are never fed back. The latest valid historical wind is the residual baseline.

## Training and metrics

AdamW, lr 0.001, weight decay 0.0001, masked Smooth L1, seed 42, FP32, batch 32, maximum 50 epochs, patience 8. The original run stopped after 9 epochs; best validation epoch 1. Complete-time partitions are 70%/15%/15%. Windows do not cross splits or gaps over 900 s. Calibration uses training only.

Historical evaluation covers 11,322 forecast origins, matching real scans within 450 s of each requested lead. Vector RMSE is `sqrt(mean((u_hat-u)^2+(v_hat-v)^2))`. Speed MAE compares magnitudes; direction uses meteorological from-direction, minimum circular difference, and a 1 m/s calm threshold. Original direction uses the three-method common non-calm mask. New local evaluation uses two methods and records that distinction.

Aggregate results include all reported leads and risk metrics. At six hours, direction error is 20.7727 degrees. Risk F1 is 0.1030 versus persistence 0.1319; improved risk detection is not claimed. Risk uses the training 95th percentile of horizontal vector-gradient norm: 0.0006461422308348119 s^-1. It is a visualization threshold, not an operational warning standard.

## Geometry and external alignment

Historical observations are Jiaozhou Airport CAPPI, not marine measurements. Grid: 33×33 at 200 m, radar-centered metres, x east/y north. Structural heights: 100/150/200/250/300 m relative to radar. Actual paired observed targets cover 100–250 m; the 300 m slot is not independently validated.

GFS 100 m above ground is one background source, not multiple observed heights. Historical alignment used airport-location proxy 36.361953 N, 120.088171 E and a UTC+8 working assumption. Exact radar installation and source timezone require provider confirmation. The public API requires explicit UTC seconds and does not infer timezone.

## Reuse

Pretrained weights assume the documented grid, heights and normalization. For another sensor, location or regime, fit training-only calibration and retrain/evaluate chronologically. The synthetic smoke test establishes execution only. Authorized local data is necessary to reproduce historical accuracy.

Code and weights use MIT. Data access and rights remain separate.
