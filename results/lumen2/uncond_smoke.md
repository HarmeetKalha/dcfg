# LUMEN2 smoke test

Verdict: **PASS**

| check | result |
|---|---|
| step-0 LUMEN2 == GenD | max |logit diff| 6.03e-04, max |embedding diff| 1.15e-04, w all 1: True -> ok |
| damaged-image stream == GenD-aug | 224 view and 256 view hashes equal in 2 slices; 25830 record-order hashes vs runs/gendaug_train/batches.log, 0 mismatches -> ok |
| 300 optimiser steps (1200 micro-steps) | main loss -1.3391 -> -1.4349, sensor loss 0.7629 -> 0.5125 (first -> last quarter), NaN steps 0, U/V abs sums 533/552 -> ok |
| resume | resumed True, batch hashes equal True, max |param diff| 4.93e-04, BN buffers (max relative diff) 4.08e-03 -> ok |
| time | 36.3 img/s -> epoch train 84.3 min + val 6.3 min; 18 epochs = **27.2 h** (limit 30 h); peak GPU 8486 MB |

## Parameters per part

| part | params |
|---|---|
| gend_trainable (LayerNorms + head, = GenD-aug) | 104,450 |
| sensor | 772,264 |
| gate (MLP + 2 LN) | 132,160 |
| ln_generator (MLP + U,V) | 837,000 |
| constant conditioning vector | 320 |
| LUMEN parts total | 1,741,744 |
