import gdsfactory as gf
import pytest


@pytest.fixture(scope="session", autouse=True)
def activate_generic_pdk() -> None:
    """Activate the generic PDK for tests that construct gdsfactory cells."""
    gf.gpdk.get_generic_pdk().activate()
