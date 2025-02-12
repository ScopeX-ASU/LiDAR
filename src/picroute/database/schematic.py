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
import ryaml
import yaml
from gdsfactory.schematic import (
    Bundle,
    Net,
    Netlist,
    # Instance,
    Placement,
)
from pydantic import AnyUrl, BaseModel, ConfigDict, Field

__all__ = [
    "Macro",
    "Instance",
    "Nets",
    "Constraints",
    "Settings",
    "CustomSchematicConfiguration",
]


class Macro(BaseModel):
    model_config = ConfigDict(extra="forbid")

    property: Optional[Dict[str, Any]] = Field(
        None, description="Properties for the component"
    )
    iloss: float = 0.1
    type: str = Field("CORE", description="Class type of the macro")
    origin: List[float] = Field(
        [0, 0], description="Coordinate of the lower-left corner"
    )
    size: List[float] = Field([1, 1], description="dimension in x and y directions")
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
        component_xl = component.size_info.west
        component_yl = component.size_info.south
        # component.x, .y means the center, but we want left lower corner
        for port in ports:
            # Port (name o1, center [-10.     -0.625], width 0.5, orientation 180.0, layer (1, 0), port_type optical)
            pins[port.name] = dict(
                pin_offset_x=float(port.center[0] - component_xl),
                pin_offset_y=float(port.center[1] - component_yl),
                pin_width=float(port.width),
                pin_orient=float(port.orientation),
                pin_layer=list(port.layer),
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
        assert len(placement) == 4
        assert type(placement[0]) == str and placement[0] in {
            "PLACED",
            "UNPLACED",
            "FIXED",
        }
        assert type(placement[1]) == list
        assert isinstance(placement[1][0], (float, int))
        assert isinstance(placement[1][1], (float, int))
        assert type(placement[2]) == str and placement[2] in {
            "N",
            "W",
            "E",
            "S",
            "FN",
            "FW",
            "FE",
            "FS",
        }
        assert type(placement[3]) == list and len(placement[3]) == 4
        return True


class dBPort(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # port attribute
    port_name: str
    center: np.array
    width: float
    orientation: float

    netName: str = None
    port_grids: list = []
    idBlk: int
    instanceName: str  # Used to locate the instance box

    gf_port: Any = None


class Blockage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idBlk: Optional[int]
    instanceName: Optional[str] = Field(
        None, description="Used to locate the instance port"
    )

    bbox: Any
    port_0: list = []
    port_90: list = []
    port_180: list = []
    port_270: list = []

    def get_ports_list(self, orientation):
        match orientation:
            case 0:
                return self.port_0
            case 90:
                return self.port_90
            case 180:
                return self.port_180
            case 270:
                return self.port_270


class Nets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    netName: str = Field(...)
    netID: int = Field(...)

    failed_count: int = 0
    enable_45: bool = True
    routed: bool = False
    wg_component: list = []
    rwguide: set = set()  # bitmap nodes' index
    routed_path: list = []
    origin_path: list = []
    rect_route: list = []

    groups: list = []
    group_name: str = None
    NetPort1: dBPort
    NetPort2: dBPort
    reverse: bool = False
    earlyaccess: bool = False

    # crossing feature
    width: int = 0.5
    wavelength: int = 143
    material: str = "WG"
    half_size: float = 5

    topology_crossing: int = 0
    maximum_crossing: int = 0
    crossing_budget: int = 100
    current_budget: int = crossing_budget
    crossing_nets: set = set()

    vionets: int = 0
    vioNets: set = set()
    # evaluation
    insertion_loss: float = 0
    wirelength: int = 0
    bending: float = 0
    crossing_num: int = 0

    settings: Optional[Dict[str, Any]] = Field(
        None, description="Settings for the component"
    )

    routing_order: int = 10000
    comp_dist: float = 100000
    distance: Optional[Dict[str, Any]] = Field(
        {"Euler": 1}, description="used for net ordering"
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
    die_area: List[List[float]] = [[0, 0], [100, 100]]
    num_instances: int = 0
    num_nets: int = 0
    num_ports: int = 0
    wg_radius: int = 5

    def update(self, **kwargs):
        for k, v in kwargs.items():
            assert hasattr(self, k)
            setattr(self, k, v)


class SchematicConfiguration(BaseModel):
    schema: AnyUrl | None = Field(None, alias="$schema")
    instances: dict[str, Instance] | None = {}
    schematic_placements: dict[str, Placement] | None = {}
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
    schema: Optional[AnyUrl] = Field(None, alias="$schema")
    settings: Optional[Settings] = None
    instances: Optional[Dict[str, Instance]] = None
    library: Optional[Dict[str, Macro]] = {}
    schematic_placements: Optional[
        Dict[str, Placement]
    ] = {}  # i.e., initial placement, not final placement solution
    nets: Any
    ports: Optional[Dict[str, str]] = None
    constraints: Optional[Dict[str, Any]] = None

    @property
    def placements(self):
        return self.schematic_placements

    def add_macro(self, component: Union[str, gf.Component]) -> None:
        new_macro = Macro(
            type="CORE",
            origin=[0, 0],
            size=list(map(float, component.size)),
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
        placement: Optional[Placement] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        macro_name, _ = self.add_macro(component)

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
            )

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


class CustomNetlist(Netlist):
    pdk: str = ""
    instances: dict[str, Instance] = Field(default_factory=dict)
    placements: dict[str, Placement] = Field(default_factory=dict)
    connections: dict[str, str] = Field(default_factory=dict)
    routes: dict[str, Bundle] = Field(default_factory=dict)
    name: str | None = None
    info: dict[str, Any] = Field(default_factory=dict)
    ports: dict[str, str] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict, exclude=True)
    nets: list[Net] = Field(default_factory=list)
    warnings: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}

    def to_yaml(self, filename) -> None:
        netlist = self.model_dump()
        with open(filename, mode="w") as f:
            yaml.dump(netlist, f, default_flow_style=None, sort_keys=False)


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

        self.pdk = pdk or gf.get_active_pdk()
        self.component_list = list(gf.get_active_pdk().cells.keys())

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
        return self._schematic

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
        self, instance_name: str, component: Union[str, gf.Component], placement=None
    ) -> None:
        self._schematic.add_instance(name=instance_name, component=component)

    def _adjust_pin_offset_orient_with_orient(
        self,
        node_orient,
        node_size_x,
        node_size_y,
        pin_offset_x,
        pin_offset_y,
        pin_orient,
    ):
        # recompute pin_offset with node orientation
        ori = pin_orient
        if node_orient == "N":  # R0:
            new_pin_orient = ori
            new_pin_offset_x = pin_offset_x
            new_pin_offset_y = pin_offset_y
        elif node_orient == "S":  # R180
            new_pin_orient = (ori + 180) % 360
            new_pin_offset_x = node_size_x - pin_offset_x
            new_pin_offset_y = node_size_y - pin_offset_y
        elif node_orient == "W":  # R90
            new_pin_orient = (ori + 90) % 360
            new_pin_offset_x = node_size_y - pin_offset_y
            new_pin_offset_y = pin_offset_x
        elif node_orient == "E":  # R270
            new_pin_orient = (ori - 90) % 360
            new_pin_offset_x = pin_offset_y
            new_pin_offset_y = node_size_x - pin_offset_x
        elif node_orient == "FN":  # MY mirror Y, cell center
            new_pin_orient = [180, 90, 0, 270][int(ori // 90)]
            new_pin_offset_x = node_size_x - pin_offset_x
            new_pin_offset_y = pin_offset_y
        elif node_orient == "FS":  # MX mirror X, cell center
            new_pin_orient = [0, 270, 180, 90][int(ori // 90)]
            new_pin_offset_x = pin_offset_x
            new_pin_offset_y = node_size_y - pin_offset_y
        elif node_orient == "FW":  # MX90 mirror X -> R90, cell center
            new_pin_orient = [90, 0, 270, 180][int(ori // 90)]
            new_pin_offset_x = pin_offset_x
            new_pin_offset_y = pin_offset_y
        elif node_orient == "FE":  # MY90 mirror Y -> R90, cell center
            new_pin_orient = [270, 180, 90, 0][int(ori // 90)]
            new_pin_offset_x = node_size_y - pin_offset_y
            new_pin_offset_y = node_size_x - pin_offset_x
        else:
            raise ValueError(f"Get {node_orient}")
        return new_pin_offset_x, new_pin_offset_y, new_pin_orient

    def load_gp(self):
        # load netlist
        path = self.path
        with open(path, "r") as f:
            # netlist = yaml.safe_load(f)
            # netlist = yaml.load(f, Loader=yaml.FullLoader)
            netlist = ryaml.load(f)

        schematic = CustomSchematicConfiguration.model_validate(netlist)
        self._schematic = schematic

        split_instance = set()
        # process instances
        blk_index = 0
        instances = {}
        for name, inst in netlist["instances"].items():
            instances[name] = Instance(
                component=inst["component"], settings=inst["settings"]
            )

            macro_type = inst["settings"]["macro_type"]
            inst_placement = inst["settings"]["placement"]
            pl, node_size_x, node_size_y = self._schematic.convert_pl_lefdef_to_gf(
                macro_type, inst_placement
            )
            self._schematic.placements[name] = pl

            if instances[name].settings.get("split", None):
                split_instance.add(name)
                del instances[name].settings["split"]

            macro = self._schematic.library[macro_type]
            size_x, size_y = macro.size
            lx, ly = inst_placement[1]
            hx, hy = lx + node_size_x, ly + node_size_y
            bbox = [lx, ly, hx, hy]

            port_0 = []
            port_90 = []
            port_180 = []
            port_270 = []

            portID = blk_index
            pins = macro.pins
            for pin_name, pin_setting in pins.items():
                new_pin_offset_x, new_pin_offset_y, new_pin_orient = (
                    self._adjust_pin_offset_orient_with_orient(
                        node_orient=inst_placement[2],
                        node_size_x=size_x,
                        node_size_y=size_y,
                        pin_offset_x=pin_setting["pin_offset_x"],
                        pin_offset_y=pin_setting["pin_offset_y"],
                        pin_orient=pin_setting["pin_orient"],
                    )
                )
                x = lx + new_pin_offset_x
                y = ly + new_pin_offset_y

                if name in split_instance:
                    portID += int(pin_name[1:]) - 1

                port_name = name + "," + pin_name
                Port = dBPort(
                    port_name=port_name,
                    center=np.array([x, y]),
                    width=pin_setting["pin_width"],
                    orientation=new_pin_orient,
                    instanceName=name,
                    idBlk=portID,
                )

                self._dbPorts[port_name] = Port
                match new_pin_orient:
                    case 0:
                        port_0.append(Port)
                    case 90:
                        port_90.append(Port)
                    case 180:
                        port_180.append(Port)
                    case 270:
                        port_270.append(Port)

            self.device_loss[name] = macro.iloss
            self._inst_blks[name] = Blockage(
                idBlk=blk_index,
                instanceName=name,
                bbox=bbox,
                port_0=port_0,
                port_90=port_90,
                port_180=port_180,
                port_270=port_270,
            )

            if name in split_instance:
                port_nums = 0
                for pin_name in macro.pins.keys():
                    if "o" in pin_name:
                        port_nums += 1
                blk_index += port_nums
            else:
                blk_index += 1

            del instances[name].settings["placement"]
            del instances[name].settings["macro_type"]

        ports = netlist.get("ports", {})
        nets = netlist.get("nets", [])
        self._schematic.instances = instances
        self._schematic.ports = ports
        self._schematic.nets = nets
        routes = {}

        netlist_conf = CustomNetlist(
            instances=schematic.instances,
            placements=schematic.placements,
            routes=routes,
            ports=schematic.ports,
        )

        layout_filename = str(self.path)[:-4] + ".layout.yml"
        netlist_conf.to_yaml(layout_filename)
        c = gf.read.from_yaml(layout_filename)
        self._gdsfactoryC = gf.Component("layout")
        self._gdsfactoryC.add_ref(c)
        self._gdsfactoryC.show()
        self.tempLayout = None

        # process nets
        cur_netid = 0
        dbNets = self._dbNets
        for unit_name, route_unit in nets.items():
            port1 = self._dbPorts[route_unit[0]]
            port2 = self._dbPorts[route_unit[1]]
            EulerDist = np.linalg.norm(port1.center - port2.center, ord=2)
            port1.netName = unit_name
            port2.netName = unit_name

            """ Get GDSFactory ports for post-processing
            """
            inst1_name, port1_id = route_unit[0].split(",")
            inst2_name, port2_id = route_unit[1].split(",")
            port1.gf_port = c.named_references[inst1_name].ports[port1_id]
            port2.gf_port = c.named_references[inst2_name].ports[port2_id]

            dbNets[unit_name] = Nets(
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
            # netlist = yaml.safe_load(f)
            netlist = ryaml.load(f)

        schematic = CustomSchematicConfiguration.model_validate(netlist)
        self._schematic = schematic

        # process instances
        instances = {
            name: Instance(component=inst["component"], settings=inst["settings"])
            for name, inst in netlist["instances"].items()
        }

        nets = netlist.get("nets", [])

        self._schematic.instances = instances
        self._schematic.nets = nets

        # process ports
        ports = netlist.get("ports", {})
        self._schematic.ports = ports

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
        constraints_cfg: Dict = dict(
            type="alignment", settings={"anchor": "left"}, objects=[]
        ),
    ):
        if constraints_cfg is not None:
            constraint_name = f"constr_{len(self._schematic.constraints)}"
            # all instances indicated by names are aligned (left, lower corner)
            self._schematic.constraints[constraint_name] = deepcopy(constraints_cfg)

    def commit(self) -> None:
        self.write_netlist()

    def get_netlist(self):
        return self._schematic.model_dump()

    def write_netlist(self) -> None:
        netlist = self.get_netlist()
        with open(self.path, mode="w") as f:
            yaml.dump(netlist, f, default_flow_style=None, sort_keys=False)

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
            c = gf.read.from_yaml(filename)
        else:
            c = gf.read.from_yaml(self.get_netlist)
        return c

    def print_benchmark(self):
        s = "\n================================== Benchmark Statistics ===================================\n\n"
        s += f"benchmark: {self.path}\n"
        s += f"{self._schematic.settings}"
        s += "\n\n===========================================================================================\n\n"

        print(s)
