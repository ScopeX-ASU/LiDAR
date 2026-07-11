"""Legacy PySpice experiment.

The SPICE fixture referenced by the original script was never included in the
repository, so this is not an executable LiDAR test.
"""

import pytest

pytest.skip(
    "manual PySpice experiment: netlist_mzi.sp is not included",
    allow_module_level=True,
)
