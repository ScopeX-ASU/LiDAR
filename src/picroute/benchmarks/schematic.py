"""Compatibility imports for historical benchmark-generation scripts.

The canonical LiDAR schematic model lives in :mod:`picroute.database.schematic`.
New code should import it from there directly.
"""

from picroute.database.schematic import *  # noqa: F403
from picroute.database.schematic import __all__ as __all__
