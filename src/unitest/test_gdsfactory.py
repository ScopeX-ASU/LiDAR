from __future__ import annotations

import gdsfactory as gf
import pytest
from gdsfactory.get_netlist import PortCenterMatcher, SmartPortMatcher, legacy_namer


def _connected_ports(netlist: dict, index: int = 0) -> set[str]:
    return set(netlist["nets"][index].values())


def test_get_netlist_cell_array() -> None:
    rows = 5
    component = gf.components.straight(length=10)
    c = gf.components.array(
        component,
        columns=1,
        rows=rows,
        column_pitch=100,
        row_pitch=100,
        add_ports=True,
    )
    netlist = c.get_netlist(on_multi_connect="ignore", instance_namer=legacy_namer)

    assert len(c.ports) == 2 * rows
    assert len(netlist["instances"]) == 1
    assert len(netlist["ports"]) == 2 * rows
    instance = next(iter(netlist["instances"].values()))
    assert instance["array"]["rows"] == rows
    assert instance["array"]["columns"] == 1


def test_get_netlist_cell_array_no_ports() -> None:
    c = gf.components.array(
        gf.components.straight(length=10),
        columns=1,
        rows=5,
        column_pitch=100,
        row_pitch=100,
        add_ports=False,
    )
    netlist = c.get_netlist(on_multi_connect="ignore")

    assert len(c.ports) == 0
    assert len(netlist["ports"]) == 0
    assert len(netlist["instances"]) == 1


def test_get_netlist_simple() -> None:
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i3 = c.add_ref(gf.components.straight(), "i3")
    i2.connect("o2", i1.ports["o1"])
    i3.dmovey(-100)

    netlist = c.get_netlist()
    assert len(netlist["nets"]) == 1
    assert _connected_ports(netlist) == {"i2,o2", "i1,o1"}


def test_get_netlist_promoted() -> None:
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i3 = c.add_ref(gf.components.straight(), "i3")
    i2.connect("o2", i1.ports["o1"])
    i3.dmovey(-100)
    c.add_port("t1", port=i1.ports["o2"])
    c.add_port("t2", port=i2.ports["o1"])
    c.add_port("t3", port=i3.ports["o1"])
    c.add_port("t4", port=i3.ports["o2"])

    netlist = c.get_netlist()
    assert len(netlist["nets"]) == 1
    assert _connected_ports(netlist) == {"i2,o2", "i1,o1"}
    assert netlist["ports"] == {
        "t1": "i1,o2",
        "t2": "i2,o1",
        "t3": "i3,o1",
        "t4": "i3,o2",
    }


def test_get_netlist_close_enough_fails() -> None:
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i2.connect("o2", i1.ports["o1"])
    i2.dmovex(1)

    assert len(c.get_netlist()["nets"]) == 0


def test_get_netlist_close_enough_orthogonal_fails() -> None:
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i2.connect("o2", i1.ports["o1"])
    i2.dmovey(0.001)

    netlist = c.get_netlist(
        port_matcher=SmartPortMatcher(position_tolerance_dbu=0.0)
    )
    assert len(netlist["nets"]) == 0


def test_get_netlist_close_enough_rotated() -> None:
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i2.connect("o2", i1.ports["o1"])
    i2.drotate(0.01)

    netlist = c.get_netlist()
    assert len(netlist["nets"]) == 1
    assert _connected_ports(netlist) == {"i2,o2", "i1,o1"}


def test_get_netlist_bad_rotation_warns() -> None:
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i2.connect("o2", i1.ports["o1"])
    i2.drotate(90)

    with pytest.warns(UserWarning):
        c.get_netlist()


def test_get_netlist_tiny() -> None:
    c = gf.Component()
    straight = gf.components.straight(length=0.002)
    i1 = c.add_ref(straight, "i1")
    i2 = c.add_ref(straight, "i2")
    i3 = c.add_ref(straight, "i3")
    i4 = c.add_ref(straight, "i4")
    i2.connect("o2", i1.ports["o1"])
    i3.connect("o2", i2.ports["o1"])
    i4.connect("o2", i3.ports["o1"])

    assert len(c.get_netlist()["nets"]) == 3


def test_get_netlist_rotated() -> None:
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i1.drotate(35)
    i2.connect("o2", i1.ports["o1"])

    netlist = c.get_netlist()
    assert len(netlist["nets"]) == 1
    assert _connected_ports(netlist) == {"i2,o2", "i1,o1"}


def test_get_netlist_electrical_simple() -> None:
    c = gf.Component()
    i1 = c.add_ref(gf.components.wire_straight(), "i1")
    i2 = c.add_ref(gf.components.wire_straight(), "i2")
    i3 = c.add_ref(gf.components.wire_straight(), "i3")
    i2.connect("e2", i1.ports["e1"])
    i3.dmovey(-100)

    netlist = c.get_netlist()
    assert len(netlist["nets"]) == 1
    assert _connected_ports(netlist) == {"i2,e2", "i1,e1"}


def test_get_netlist_electrical_rotated_joint() -> None:
    c = gf.Component()
    i1 = c.add_ref(gf.components.wire_straight(), "i1")
    i2 = c.add_ref(gf.components.wire_straight(), "i2")
    i2.connect("e2", i1.ports["e1"])
    i2.drotate(45)

    netlist = c.get_netlist()
    assert len(netlist["nets"]) == 1
    assert _connected_ports(netlist) == {"i2,e2", "i1,e1"}


def test_get_netlist_electrical_different_widths() -> None:
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(width=1, cross_section="metal1"), "i1")
    i2 = c.add_ref(gf.components.straight(width=10, cross_section="metal1"), "i2")
    i2.connect("e2", i1.ports["e1"], allow_width_mismatch=True)

    netlist = c.get_netlist(port_matcher=PortCenterMatcher())
    assert len(netlist["nets"]) == 1
    assert _connected_ports(netlist) == {"i2,e2", "i1,e1"}


def test_get_netlist_transformed() -> None:
    c = gf.Component()
    i1 = c.add_ref(gf.components.straight(), "i1")
    i2 = c.add_ref(gf.components.straight(), "i2")
    i1.drotate(35)
    i2.connect("o2", i1.ports["o1"])

    netlist = c.get_netlist()
    assert len(netlist["nets"]) == 1
    assert _connected_ports(netlist) == {"i2,o2", "i1,o1"}
