"""Models supported by ActionShap.

``static`` is the legacy clustering model. ``profile`` is the recommendation
model used by the revised recommendation-only specification.
"""

from .profile import ProfileAggregationModel

__all__ = ["ProfileAggregationModel"]
