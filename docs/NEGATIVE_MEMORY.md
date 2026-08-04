# Sandwiched Negative Memory

Version 5.2 adds a sidecar model that learns what happens when BTC approaches or crosses a confirmed support or resistance level.

## Objective

For each side and trade horizon (3h, 6h, 12h), the model estimates:

- probability that the active level breaks in the modeled direction;
- probability that the setup fails to produce positive stress-net profit;
- whether the current discretized market fingerprint matches recurring losing history.

Support heads are trained on downside approaches and support breakdowns. Resistance heads are trained on upside approaches and resistance breakouts. Inputs include volume anomalies, volatility, momentum, candle quality, trend regime, level strength, level age, distance in ATR, triangle context, and recent news features.

## Sandwich

The runtime guard uses three layers:

1. **Front Bloom memory** — recurring fingerprints with a high historical no-profit rate.
2. **Learned middle** — separate break-probability and no-profit classifiers.
3. **Backup Bloom memory** — hard negatives that the learned middle incorrectly accepted during chronological calibration.

A Bloom hit is a conservative veto. False positives can reject an opportunity, but cannot directly authorize a trade.

## Hard-negative mining

Every non-profitable boundary encounter is a negative example. Negatives that the first no-profit classifier assigns a low risk are up-weighted and the classifier is fitted again. This focuses learning on patterns that look attractive but historically failed after stress costs.

## Chronological validation

Each side/horizon head uses:

- 70% chronological training;
- 15% calibration and policy selection;
- 15% locked holdout.

The policy selects minimum break probability and maximum no-profit probability only on calibration data. A head is qualified only when the locked holdout has enough accepted samples, positive mean stress-net return, acceptable profitable-break rate, and limited bad acceptance.

## Promotion rule

A challenger is promoted only when at least one direction/horizon passes both:

- the existing locked economic breakout gate;
- the matching boundary-memory holdout gate.

LONG requires a qualified RESISTANCE head. SHORT requires a qualified SUPPORT head.

## Runtime safety

If the sidecar is missing, belongs to another model ID, is unqualified, reports a Bloom hit, predicts insufficient break probability, or predicts excessive no-profit risk, the trade action is changed to `WAIT`. The public next-candle forecast remains available for research.
