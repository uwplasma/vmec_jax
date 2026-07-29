"""Small compatibility helpers shared by VMEC netCDF writers."""

from __future__ import annotations

import warnings
from typing import Any

_NUMPY25_SHAPE_WARNING = (
    r"Setting the shape on a NumPy array has been deprecated in NumPy 2\.5\."
)


def assign_array(variable: Any, key: Any, value: Any) -> None:
    """Assign an array while isolating netCDF4's NumPy 2.5 warning.

    netCDF4 1.7.4 reshapes an internal view with ``data.shape = ...`` for
    every non-scalar assignment. NumPy 2.5 deprecates that internal
    operation even though the write and file layout are valid. Keep the
    suppression exact and local so unrelated deprecations remain visible.
    """

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=_NUMPY25_SHAPE_WARNING,
            category=DeprecationWarning,
        )
        variable[key] = value
