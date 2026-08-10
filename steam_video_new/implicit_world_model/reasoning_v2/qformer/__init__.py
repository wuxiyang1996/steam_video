"""Four-slot, node-aligned Q-Former baseline.

The persistence contracts remain importable without PyTorch. Neural components
are lazy-loaded so existing reasoning-v2 runtimes keep their lightweight
dependency boundary.
"""

from .contracts import (
    FEATURE_STORE_SCHEMA,
    QF1_CACHE_SCHEMA,
    SLOT_NAMES,
    FeatureMatrixSpec,
    FeatureRow,
    FeatureStoreManifest,
    QF1CacheManifest,
)
from .feature_store import FourSlotFeatureStore, write_feature_store
from .qf1_cache import QF1Cache

__all__ = [
    "FEATURE_STORE_SCHEMA",
    "QF1_CACHE_SCHEMA",
    "SLOT_NAMES",
    "FeatureMatrixSpec",
    "FeatureRow",
    "FeatureStoreManifest",
    "FourSlotFeatureStore",
    "QF1CacheManifest",
    "QF1Cache",
    "write_feature_store",
]
