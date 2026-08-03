# Fail-safe v2

A paper action is blocked when any applicable condition is present:

- incompatible v1 model artifact;
- no newly confirmed market event;
- Event ID already traded;
- selected horizon not qualified;
- low direction confidence;
- low tradeability probability;
- horizon disagreement;
- expected edge below stress execution cost and profit buffer;
- stale quote, provider mismatch or candle gap;
- excessive ATR volatility;
- configured news shock;
- daily signal limit or cooldown;
- stale model.

UP/DOWN remains visible even when action is WAIT. Live order execution is not implemented.
