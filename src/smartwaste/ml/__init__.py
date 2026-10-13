"""Phase 3 models: fill-level forecasting and sensor anomaly detection.

**Nothing in this package reads ground truth.** Forecast features are built only from past noisy
sensor estimates plus operational records (area type, weekday, festival flag); the anomaly
detectors are handed readings and never the planted-fault label file. See
:mod:`smartwaste.ml.forecast` and :mod:`smartwaste.ml.anomaly`.

**scikit-learn is imported lazily**, inside the functions that actually fit a model, never at
module import time. That is what lets the naive forecast baselines and the rule-based anomaly
detector -- both pure stdlib -- run in the ``unit`` / ``sim`` / ``routing`` / ``slow`` test
chunks without the optional ``ml`` extra installed.
"""

from __future__ import annotations
