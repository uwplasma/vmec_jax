"""Compatibility shim for :mod:`vmex.core.virtual_casing`.

Use :class:`vmex.PlasmaVacuumInterface` for new code. This module name predates
the distinction between prescribed-interface virtual casing and a true
free-boundary equilibrium solve.
"""

from .virtual_casing import *  # noqa: F403
from .virtual_casing import __all__ as __all__
