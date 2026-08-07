# Online support policy

The online price learner remains subordinate to the batch champion.

## Modes

- `BATCH_ONLY`: online output has no influence.
- `SUPPORT_ONLY`: a small direction-only weight is allowed when the online and batch directions agree and the online rolling accuracy remains within a small tolerance of the batch model.
- `EVIDENCE_WEIGHTED`: the existing strict improvement gates allow a larger blend only when the online learner is measurably better on Brier score and direction accuracy.

## Support-only safety

The support-only path:

- never changes the predicted side when online disagrees with batch;
- uses only direction, not the return estimate;
- caps the online probability before blending to avoid overconfident shadow outputs;
- is limited to 5% by default;
- requires a mature rolling evaluation window and at least 50% rolling direction accuracy.

This lets recent adaptation nudge confidence while preserving the batch champion as the dominant model.
