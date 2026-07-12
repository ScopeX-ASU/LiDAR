"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2023-07-17 14:25:34
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-02-05 01:08:17
"""

import os
from copy import deepcopy

# from math import e
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import gdsfactory as gf
import networkx as nx
import numpy as np
import yaml
from gdsfactory.schematic import Placement
from pydantic import AnyUrl, BaseModel, ConfigDict, Field

__all__ = [
    "Macro",
    "Instance",
    "EndpointState",
    "RouteState",
    "Constraints",
    "Settings",
    "CustomSchematicConfiguration",
    "CustomSchematic",
]


class Macro(BaseModel):
    model_config = ConfigDict(extra="forbid")

    property: Optional[Dict[str, Any]] = Field(
        None, description="Properties for the component"
    )
    iloss: float = 0.1
    type: str = Field("CORE", description="Class type of the macro")
    origin: List[float] = Field(
        default_factory=lambda: [0, 0],
        description="Coordinate of the lower-left corner",
    )
    size: List[float] = Field(
        default_factory=lambda: [1, 1], description="dimension in x and y directions"
    )
    site: str = "core"
    pins: Optional[Dict[str, Any]] = Field(
        None, description="pin difinitions as a dictionary"
    )

    @classmethod
    def extract_pins(cls, component: gf.Component) -> Dict[str, str]:
        """
        PIN ck DIRECTION INPUT ;
            PORT
            LAYER metal1 ;
            RECT 0.45 0.500 0.55 1.500 ;
            END
        END ck
        """
        pins = dict()
        ports = component.get_ports_list(port_type="optical")
        component_xl = component.dsize_info.west
        component_yl = component.dsize_info.south
        # component.x, .y means the center, but we want left lower corner
        for port in ports:
            # Port (name o1, center [-10.     -0.625], width 0.5, orientation 180.0, layer (1, 0), port_type optical)
            pins[port.name] = dict(
                pin_offset_x=float(port.dcenter[0] - component_xl),
                pin_offset_y=float(port.dcenter[1] - component_yl),
                pin_width=float(port.dwidth),
                pin_orient=float(port.orientation),
                pin_layer=[port.layer_info.layer, port.layer_info.datatype],
            )
        return pins


class Instance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    component: str = "straight"
    settings: Optional[Dict[str, Any]] = Field(
        None, description="Settings for the component"
    )

    @staticmethod
    def verify_placement(placement):
        # PLACEMENT_STATUS
        # LOCATION [x, y]
        # ORIENTATION
        # HALO [l, d, r, u]
        if len(placement) != 4:
            raise ValueError(
                "Placement requires status, location, orientation, and halo"
            )
        if not isinstance(placement[0], str) or placement[0] not in {
            "PLACED",
            "UNPLACED",
            "FIXED",
        }:
            raise ValueError(f"Invalid placement status {placement[0]!r}")
        if not (
            isinstance(placement[1], list)
            and len(placement[1]) == 2
            and all(isinstance(value, (float, int)) for value in placement[1])
        ):
            raise ValueError("Placement location must contain two numbers")
        if not isinstance(placement[2], str) or placement[2] not in {
            "N",
            "W",
            "E",
            "S",
            "FN",
            "FW",
            "FE",
            "FS",
        }:
            raise ValueError(f"Invalid placement orientation {placement[2]!r}")
        if not isinstance(placement[3], list) or len(placement[3]) != 4:
            raise ValueError("Placement halo must contain four values")
        return True


class EndpointState(BaseModel):
    """LiDAR routing state attached to one native transformed port."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    port_name: str
    port: gf.Port

    netName: str | None = None
    port_grids: list = Field(default_factory=list)
    idBlk: int
    instanceName: str  # Used to locate the instance box

    @property
    def center(self) -> np.ndarray:
        return np.asarray(self.port.dcenter, dtype=float)

    @property
    def width(self) -> float:
        return float(self.port.dwidth)

    @property
    def orientation(self) -> float:
        return float(round(self.port.orientation) % 360)


class Blockage(BaseModel):
    """Routing blockage backed by a native transformed component reference."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    idBlk: Optional[int]
    instanceName: Optional[str] = Field(
        None, description="Used to locate the instance port"
    )

    reference: gf.ComponentReference
    ports_by_orientation: dict[int, list[EndpointState]] = Field(default_factory=dict)

    @property
    def bbox(self):
        return self.reference.dbbox()

    def get_ports_list(self, orientation):
        return self.ports_by_orientation.get(orientation, [])


class RouteState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    netName: str = Field(...)
    netID: int = Field(...)

    failed_count: int = 0
    enable_45: bool = True
    routed: bool = False
    wg_component: list = Field(default_factory=list)
    rwguide: set = Field(default_factory=set)  # bitmap nodes' index
    routed_path: list = Field(default_factory=list)
    origin_path: list = Field(default_factory=list)
    rect_route: list = Field(default_factory=list)

    groups: list = Field(default_factory=list)
    group_name: str = None
    NetPort1: EndpointState
    NetPort2: EndpointState
    reverse: bool = False
    earlyaccess: bool = False

    # crossing feature
    width: float = 0.5
    wavelength: int = 143
    material: str = "WG"
    half_size: float = 5

    topology_crossing: int = 0
    maximum_crossing: int = 0
    crossing_budget: int = 100
    current_budget: int = crossing_budget
    crossing_nets: set = Field(default_factory=set)

    vionets: int = 0
    vioNets: set = Field(default_factory=set)
    # evaluation
    insertion_loss: float = 0
    wirelength: float = 0
    bending: float = 0
    crossing_num: int = 0

    settings: Optional[Dict[str, Any]] = Field(
        None, description="Settings for the component"
    )

    routing_order: int = 10000
    comp_dist: float = 100000
    distance: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {"Euler": 1}, description="used for net ordering"
    )

    def clearNet(self, cirdb, routed=False, increase=True):
        self.routed = routed
        self.wg_component.clear()
        self.rwguide.clear()
        self.routed_path.clear()
        self.bending = 0
        for net_name in self.crossing_nets:
            net = cirdb.dbNets[net_name]
            if self.netName in net.crossing_nets:
                net.crossing_nets.remove(self.netName)
                # net.current_budget += 1
        self.crossing_nets.clear()
        self.insertion_loss = 0
        self.wirelength = 0
        self.bending = 0
        self.crossing_num = 0
        self.vionets = 0
        self.vioNets = set()

        if increase:
            self.failed_count += 1
        if self.failed_count > 4:
            self.enable_45 = False

    def backup_net(self):
        crossing_nets = deepcopy(self.crossing_nets)
        return (self.failed_count, crossing_nets, self.crossing_num)

    @property
    def port1(self):
        return self.NetPort1

    @property
    def port2(self):
        return self.NetPort2

    @property
    def instance1(self):
        return self.NetPort1.instanceName

    @property
    def instance2(self):
        return self.NetPort2.instanceName

    ### This __eq__ and __lt__ function is to make the Net object comparable, so the priority queue knows how to sort nodes.
    def __eq__(self, other):
        return (
            self.comp_dist == other.comp_dist
            and self.routing_order == other.routing_order
        )

    def __lt__(self, other):
        return (self.comp_dist < other.comp_dist) or (
            (self.comp_dist == other.comp_dist)
            and (self.routing_order < other.routing_order)
        )

    def crossing_check(self, net2, straight_length):
        if (
            self.width == net2.width
            and self.wavelength == net2.wavelength
            and self.material == net2.material
            and straight_length > self.half_size
        ):
            return True, self.half_size
        return False, None


class Constraints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = Field(None, description="Constraint type, now supports alignment")
    settings: Dict = Field(None, description="Constraint settings")
    objects: List = Field(
        None,
        description="Apply constraints to the objects, can be instance names, or net names",
    )


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = "1.0"
    design: str = "default_design"
    units_distance_microns: float = 1  # 1 unit = 1 micro
    die_area: List[List[float]] = Field(default_factory=lambda: [[0, 0], [100, 100]])
    num_instances: int = 0
    num_nets: int = 0
    num_ports: int = 0
    wg_radius: int = 5

    def update(self, **kwargs):
        for k, v in kwargs.items():
            assert hasattr(self, k)
            setattr(self, k, v)


class SchematicConfiguration(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    schema_: AnyUrl | None = Field(None, alias="$schema")
    instances: dict[str, Instance] | None = Field(default_factory=dict)
    schematic_placements: dict[str, Placement] | None = Field(default_factory=dict)
    nets: list[list[str]] | None = None
    ports: dict[str, str] | None = None
    schema_version: int = Field(
        default=1, description="The version of the YAML syntax used."
    )

    @property
    def placements(self):
        return self.schematic_placements

    def add_instance(
        self,
        name: str,
        component: str | gf.Component,
        placement: Placement | None = None,
        settings: dict[str, Any] | None = None,
    ) -> None:
        if isinstance(component, gf.Component):
            component_name = component.function_name
            component_settings = component.settings.model_dump()
            if settings:
                component_settings = component_settings | settings
            self.instances[name] = Instance(
                component=component_name, settings=component_settings
            )
        else:
            if not settings:
                settings = {}
            self.instances[name] = Instance(component=component, settings=settings)
        if name not in self.placements:
            if not placement:
                placement = Placement()
            self.placements[name] = placement


class CustomSchematicConfiguration(SchematicConfiguration):
    settings: Optional[Settings] = None
    instances: Optional[Dict[str, Instance]] = None
    library: Optional[Dict[str, Macro]] = Field(default_factory=dict)
    schematic_placements: Optional[Dict[str, Placement]] = Field(default_factory=dict)
    nets: Any
    ports: Optional[Dict[str, str]] = None
    constraints: Optional[Dict[str, Any]] = None

    @property
    def placements(self):
        return self.schematic_placements

    def add_macro(
        self, component: Union[str, gf.Component], iloss: float = 0.1
    ) -> tuple[str, Macro]:
        new_macro = Macro(
            iloss=iloss,
            type="CORE",
            origin=[0, 0],
            size=[float(component.dxsize), float(component.dysize)],
            site="core",
            pins=Macro.extract_pins(component),
        )
        for macro_name, macro in self.library.items():
            if (
                component.name in macro_name
            ):  # match the type, but need further check for pcell
                if macro == new_macro:
                    # find the same macro, then skip
                    return macro_name, new_macro
        new_macro_name = f"m_{component.name}_{len(self.library)}"
        self.library[new_macro_name] = new_macro
        return new_macro_name, new_macro

    def add_instance(
        self,
        name: str,
        component: Union[str, gf.Component],
        iloss: float = 0.1,
        placement: Optional[Placement] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        macro_name, _ = self.add_macro(component, iloss=iloss)

        if isinstance(component, gf.Component):
            # component_name = component.settings.function_name
            component_name = component.function_name
            component_settings = component.settings.model_dump()
            # component_settings = component.settings.changed
            if settings:
                component_settings = component_settings | settings
            component_settings = component_settings | {"macro_type": macro_name}
            self.instances[name] = Instance(
                component=component_name, settings=component_settings
            )
        else:
            if not settings:
                settings = {}
            self.instances[name] = Instance(component=component, settings=settings)
        # record and add macro in the library

        # physical placement of instances
        self.update_placement(name, placement)

        # schematic placement
        if name not in self.placements:
            if not placement:
                placement = Placement()
            self.placements[name] = placement
        self.settings.num_instances += 1

    def update_placement(self, name: str, placement: List[Any] = None):
        if placement is None:
            placement = [
                "UNPLACED",
                [0, 0],
                "N",
                [0, 0, 0, 0],
            ]
        if Instance.verify_placement(placement):
            node = self.instances[name]
            node.settings["placement"] = placement
            self.placements[name] = self.convert_pl_lefdef_to_gf(
                node.settings["macro_type"], placement
            )[0]

    def convert_pl_lefdef_to_gf(self, macro_type: str, placement_sol: dict):
        placement = Placement()
        placement.x = float(placement_sol[1][0])
        placement.y = float(placement_sol[1][1])
        orient = placement_sol[2]
        placement.port = (
            "sw"  # use port to control the origin, south-west means lower-left corner
        )
        macro = self.library[macro_type]
        node_size_x, node_size_y = macro.size
        if orient in {"W", "E", "FW", "FE"}:
            node_size_x, node_size_y = node_size_y, node_size_x
        if orient == "N":
            pass
        elif orient == "S":
            placement.rotation = 180
            # we need to adjust x, adding node_size_x
            # placement.dx = node_size_x
            # placement.dy = node_size_y
        elif orient == "W":
            placement.rotation = 90
            # we need to adjust x, adding node_size_x
            # placement.dx = node_size_x
        elif orient == "E":
            placement.rotation = 270
            # we need to adjust x, adding node_size_x
            # placement.dy = node_size_y
        elif orient == "FN":
            placement.mirror = True
        elif orient == "FS":  # flip X, vertical flip
            placement.mirror = True
            # after mirror_x horizontally along left edge, the left_lower corner is still [x, y]
            # after 180 rotation, [x, y] becomes the upper-right corner
            placement.rotation = 180
            # we need to adjust x, adding node_size_x
            # placement.dx = node_size_x
            # placement.dy = node_size_y
        elif orient == "FW":
            placement.mirror = True
            # placement.dy = node_size_y
        elif orient == "FE":
            placement.mirror = True
            # placement.dx = node_size_x
        else:
            raise ValueError

        return placement, node_size_x, node_size_y


class CustomSchematic(object):
    def __init__(
        self, filename: Union[str, Path], pdk: Optional[gf.Pdk] = None
    ) -> None:
        """An interactive Schematic editor, meant to be used from a Jupyter Notebook.

        Args:
            filename: the filename or path to use for the input/output schematic
            pdk: the PDK to use (uses the current active PDK if None)
        """
        filepath = filename if isinstance(filename, Path) else Path(filename)
        self.path = filepath

        if pdk is None:
            try:
                pdk = gf.get_active_pdk()
            except ValueError:
                pdk = gf.gpdk.get_generic_pdk()
                pdk.activate()
        else:
            pdk.activate()
        self.pdk = pdk
        self.component_list = list(pdk.cells)

        self.on_instance_added = []
        self.on_instance_removed = []
        self.on_settings_updated = []
        self.on_nets_modified = []
        self._notebook_handle = None
        self._connected_ports = {}

        self._inst_blks = {}
        self._dbNets = {}
        self._dbPorts = {}
        self._group_Nets = {}
        self._abnormal_Nets = {}
        self._gdsfactoryC = (
            None  # high-level gdsfactory component, which is used for visualization
        )
        self.device_loss = {}
        self._native_schematic = gf.Schematic()
        self.connectivity_report = None

        if 0 and filepath.is_file():
            self.load_netlist()
        else:
            self._schematic = CustomSchematicConfiguration(
                settings=Settings(),
                library={},
                instances={},
                schematic_placements={},
                nets={},
                ports={},
                constraints={},
            )

    @property
    def schematic(self):
        """Return the GP compatibility model used by benchmark generators."""
        return self._schematic

    @property
    def native_schematic(self) -> gf.Schematic:
        """Return the canonical gdsfactory/kfactory schematic."""
        return self._build_native_schematic()

    @property
    def layout(self):
        return self._gdsfactoryC

    @layout.setter
    def layout(self, value):
        self._gdsfactoryC = value

    @property
    def initBlk(self):
        return self._inst_blks

    @property
    def dbNets(self):
        return self._dbNets

    @property
    def abNets(self):
        return self._abnormal_Nets

    def update_settings(self, **kwargs):
        self._schematic.settings.update(**kwargs)

    def add_instance(
        self,
        instance_name: str,
        component: Union[str, gf.Component],
        iloss: float = 0.1,
        placement=None,
    ) -> None:
        self._schematic.add_instance(
            name=instance_name, component=component, iloss=iloss, placement=placement
        )

    @staticmethod
    def _factory_settings(instance: Instance) -> dict[str, Any]:
        """Return only settings accepted by the component factory."""
        settings = dict(instance.settings or {})
        for key in ("macro_type", "placement", "split"):
            settings.pop(key, None)
        return settings

    def _build_native_schematic(self) -> gf.Schematic:
        """Translate the GP interchange model into a native micron schematic."""
        gp = self._schematic
        native = gf.Schematic(
            name=gp.settings.design,
            info={"gp_settings": gp.settings.model_dump()},
        )

        for name, instance in gp.instances.items():
            native_instance = native.create_inst(
                name=name,
                component=instance.component,
                settings=self._factory_settings(instance),
            )
            placement_data = (instance.settings or {}).get("placement")
            if placement_data is None:
                placement_data = ["UNPLACED", [0, 0], "N", [0, 0, 0, 0]]
            placement, _, _ = gp.convert_pl_lefdef_to_gf(
                (instance.settings or {})["macro_type"], placement_data
            )
            native_instance.place(
                x=float(placement.x or 0),
                y=float(placement.y or 0),
                orientation=float(placement.rotation),
                mirror=bool(placement.mirror),
                anchor={"x": "left", "y": "bottom"},
            )

        for endpoints in gp.nets.values():
            instance1, port1 = endpoints[0].split(",", maxsplit=1)
            instance2, port2 = endpoints[1].split(",", maxsplit=1)
            native.connect(
                native.instances[instance1].ports[port1],
                native.instances[instance2].ports[port2],
                virtual=True,
            )

        self._native_schematic = native
        return native

    def _materialize_native_schematic(self) -> gf.Component:
        """Create the placed component and enforce transformed GP lower-left anchors."""
        native = self._build_native_schematic()
        component = native.create_cell(
            output_type=gf.Component,
            factories=self.pdk.cells,
        )

        # GP/LEF placements locate the lower-left corner after transformation.
        # DSchematic fixed anchors refer to the pre-rotation bbox, so normalize the
        # resulting native references to the GP convention.
        for name, instance in self._schematic.instances.items():
            placement_data = (instance.settings or {}).get("placement")
            if placement_data is None:
                continue
            reference = component.insts[name]
            reference.dxmin = float(placement_data[1][0])
            reference.dymin = float(placement_data[1][1])
        return component

    def load_gp(self):
        # load netlist
        path = self.path
        with open(path, "r") as f:
            netlist = yaml.safe_load(f)

        schematic = CustomSchematicConfiguration.model_validate(netlist)
        self._schematic = schematic

        split_instance = set()
        instance_macros = {}
        for name, instance in schematic.instances.items():
            settings = instance.settings or {}
            macro_type = settings["macro_type"]
            instance_macros[name] = self._schematic.library[macro_type]
            if settings.get("split", False):
                split_instance.add(name)

        ports = netlist.get("ports", {})
        nets = netlist.get("nets", {})
        self._schematic.ports = ports
        self._schematic.nets = nets

        c = self._materialize_native_schematic()
        self._gdsfactoryC = c
        self.tempLayout = None

        # Use GDSFactory/KFactory as the source of truth for transformed
        # bounding boxes and ports instead of duplicating transform algebra.
        blk_index = 0
        for reference in c.insts:
            name = reference.name
            orientation_ports = {0: [], 90: [], 180: [], 270: []}
            optical_ports = list(reference.ports.filter(port_type="optical"))

            for port in optical_ports:
                orientation = int(round(port.orientation)) % 360
                if orientation not in orientation_ports:
                    raise ValueError(
                        f"LiDAR grid access requires cardinal ports; "
                        f"got {orientation} degrees for {name},{port.name}"
                    )
                port_id = blk_index
                if name in split_instance:
                    port_id += int(port.name[1:]) - 1
                port_name = f"{name},{port.name}"
                db_port = EndpointState(
                    port_name=port_name,
                    port=port,
                    instanceName=name,
                    idBlk=port_id,
                )
                self._dbPorts[port_name] = db_port
                orientation_ports[orientation].append(db_port)

            self.device_loss[name] = instance_macros[name].iloss
            self._inst_blks[name] = Blockage(
                idBlk=blk_index,
                instanceName=name,
                reference=reference,
                ports_by_orientation=orientation_ports,
            )
            blk_index += len(optical_ports) if name in split_instance else 1

        # process nets
        cur_netid = 0
        dbNets = self._dbNets
        for unit_name, route_unit in nets.items():
            port1 = self._dbPorts[route_unit[0]]
            port2 = self._dbPorts[route_unit[1]]
            EulerDist = np.linalg.norm(port1.center - port2.center, ord=2)
            port1.netName = unit_name
            port2.netName = unit_name

            dbNets[unit_name] = RouteState(
                netName=unit_name,
                netID=cur_netid,
                # group_name=inst1_name,
                NetPort1=port1,
                NetPort2=port2,
                comp_dist=int(EulerDist / 1000),  # inter group priority
                routing_order=int(EulerDist),  # intra group priority
                distance={"Euler": EulerDist},
            )
            cur_netid += 1

        # self.device_loss = {
        #     "mmi": 0.1,
        #     "straight_heater_metal_undercut": 0.05,
        #     "grating_coupler_elliptical_lumerical": 2,
        #     "mmi1x2": 0.3,
        #     "mzi": 1.2,
        #     "straight": 0.03,
        #     "ring_single_pn": 1,
        # }

        self._netlist_graph = nx.DiGraph()
        G = self._netlist_graph
        self.topology_orders = []

        for net in self._dbNets.values():
            G.add_edge(
                net.NetPort1.idBlk,
                net.NetPort2.idBlk,
                weight=net.distance["Euler"],
                label=net.netName,
            )

        def find_source_nodes(graph):
            return [node for node in graph.nodes() if graph.in_degree(node) == 0]

        def find_end_nodes(graph):
            return [node for node in graph.nodes() if graph.out_degree(node) == 0]

        def find_graph_neighbors(graph, source):
            visited = set()
            current_level = set(source)
            while current_level:
                nets = []
                # print(f"{sorted(current_level)}")
                next_level = set()
                for node in current_level:
                    for neighbor in graph.successors(node):
                        next_level.add(neighbor)
                        net = graph[node][neighbor]["label"]
                        if net not in visited:
                            nets.append(net)
                            visited.add(net)
                if len(nets) != 0:
                    self.topology_orders.append(nets)
                current_level = next_level

        sources = find_source_nodes(G)
        find_graph_neighbors(G, sources)

    def load_netlist(self, path=None) -> None:
        path = path or self.path
        with open(path, "r") as f:
            netlist = yaml.safe_load(f)

        schematic = CustomSchematicConfiguration.model_validate(netlist)
        self._schematic = schematic
        self._build_native_schematic()

    def add_net(self, inst1, port1, inst2, port2):
        p1 = f"{inst1},{port1}"
        p2 = f"{inst2},{port2}"
        if p1 in self._connected_ports:
            if self._connected_ports[p1] == p2:
                return
            current_port = self._connected_ports[p1]
            raise ValueError(
                f"{p1} is already connected to {current_port}. Can't connect to {p2}"
            )
        self._connected_ports[p1] = p2
        self._connected_ports[p2] = p1
        old_nets = self._schematic.nets.copy()
        net_name = f"n_{len(old_nets)}"
        self._schematic.nets[net_name] = [p1, p2]

        self._schematic.settings.num_nets += 1
        print(f"add nets {self._schematic.settings.num_nets}: {[p1, p2]}")
        assert self._schematic.settings.num_nets == len(self._schematic.nets)

    def update_placement(self, name: str, placement: List[Any]):
        self._schematic.update_placement(name, placement)

    def add_constraints(
        self,
        constraints_cfg: Dict | None = None,
    ):
        if constraints_cfg is None:
            constraints_cfg = {
                "type": "alignment",
                "settings": {"anchor": "left"},
                "objects": [],
            }
        if constraints_cfg is not None:
            constraint_name = f"constr_{len(self._schematic.constraints)}"
            # all instances indicated by names are aligned (left, lower corner)
            self._schematic.constraints[constraint_name] = deepcopy(constraints_cfg)

    def commit(self) -> None:
        self._build_native_schematic()
        self.write_netlist()

    def get_netlist(self):
        return self._schematic.model_dump()

    def write_netlist(self) -> None:
        netlist = self.get_netlist()
        with open(self.path, mode="w") as f:
            yaml.safe_dump(netlist, f, default_flow_style=None, sort_keys=False)

    def plot_netlist(
        self,
        with_labels: bool = True,
        font_weight: str = "normal",
    ):
        """Plots a netlist graph with networkx.

        Args:
            with_labels: add label to each node.
            font_weight: normal, bold.
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        plt.figure()
        nets = self._schematic.nets
        placements = self._schematic.placements
        G = nx.Graph()
        G.add_edges_from(
            [(net[0].split(",")[0], net[1].split(",")[0]) for net in nets.values()]
        )
        pos = {k: (v.x, v.y) for k, v in placements.items()}
        labels = {k: k for k in placements.keys()}

        for node, placement in placements.items():
            if not G.has_node(
                node
            ):  # Check if the node is already in the graph (from connections), to avoid duplication.
                G.add_node(node)
                pos[node] = (placement.x, placement.y)

        nx.draw(
            G,
            with_labels=with_labels,
            font_weight=font_weight,
            labels=labels,
            pos=pos,
        )
        return G

    def to_component(self, filename: str | None = None) -> gf.Component:
        if filename is not None and os.path.exists(filename):
            with open(filename, "r") as stream:
                self._schematic = CustomSchematicConfiguration.model_validate(
                    yaml.safe_load(stream)
                )
        return self._materialize_native_schematic()

    def verify_connectivity(self):
        """Run kfactory's native optical connectivity checks on the layout."""
        if self.layout is None:
            raise RuntimeError("Cannot verify connectivity before materializing layout")
        self.connectivity_report = self.layout.connectivity_check(
            port_types=["optical"],
            recursive=True,
            add_cell_ports=True,
        )
        return self.connectivity_report

    def print_benchmark(self):
        s = "\n================================== Benchmark Statistics ===================================\n\n"
        s += f"benchmark: {self.path}\n"
        s += f"{self._schematic.settings}"
        s += "\n\n===========================================================================================\n\n"

        print(s)
