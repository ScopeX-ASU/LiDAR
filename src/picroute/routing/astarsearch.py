import math
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from heapq import heappop, heappush
from itertools import count
from typing import List, Optional, Self, Tuple

import gdsfactory as gf
import numpy as np
import shapely.geometry as sg

from picroute.database.schematic import CustomSchematic, RouteState
from picroute.drc.drcmanager import DrcManager
from picroute.utils.smooth import prepare_waypoints

from .drgridroute import DrGridRoute
from .utils import EulerDist, customH, manhattanDH

type Transition = tuple[int, int, int, str]


def _rotate_transition(transition: Transition, quarter_turns: int) -> Transition:
    x, y, orientation, step_type = transition
    for _ in range(quarter_turns % 4):
        x, y = -y, x
    return x, y, (orientation + 90 * quarter_turns) % 360, step_type


def _build_transitions(
    grid_radius: int,
    bend45_1: int,
    bend45_2: int,
    *,
    enable_45: bool,
) -> dict[int, list[Transition]]:
    bend_sum = bend45_1 + bend45_2
    cardinal = [
        (grid_radius, -grid_radius, 270, "bend_90"),
        (1, 0, 0, "straight_0"),
        (grid_radius, grid_radius, 90, "bend_90"),
    ]
    diagonal = [
        (bend_sum, bend45_2, 0, "bend_45_2"),
        (1, 1, 45, "straight_45"),
        (bend45_2, bend_sum, 90, "bend_45_2"),
    ]
    transitions = {
        90 * turns: [_rotate_transition(step, turns) for step in cardinal]
        for turns in range(4)
    }
    transitions.update(
        {
            45 + 90 * turns: [_rotate_transition(step, turns) for step in diagonal]
            for turns in range(4)
        }
    )

    if enable_45:
        extra = [
            (bend_sum, -bend45_2, 315, "bend_45_1"),
            (bend_sum, bend45_2, 45, "bend_45_1"),
        ]
        for turns in range(4):
            rotated = [_rotate_transition(step, turns) for step in extra]
            if turns == 1:
                rotated.reverse()
            transitions[90 * turns].extend(rotated)
    return transitions


@dataclass(eq=False, slots=True)
class GridAstarNode:
    pos: tuple[int, int, int] = (0, 0, 0)
    cost_g: float = float("inf")
    cost_f: float = float("inf")
    visited: bool = False
    parent: Self | None = None
    neighbors: dict[Self, str] = field(default_factory=dict)
    crossing_budget: int = 0
    crossing_net: str | None = None
    violated_nets: set[str] = field(default_factory=set)
    is_violated: bool = False
    straight_count: int = 0

    def reset(self) -> None:
        self.cost_g = float("inf")
        self.cost_f = float("inf")
        self.visited = False
        self.parent = None

    def add_neighbor(self, neighbor: Self, step_type: str) -> None:
        self.neighbors[neighbor] = step_type

    def __eq__(self, other):
        if not isinstance(other, GridAstarNode):
            return NotImplemented
        return self.pos == other.pos

    def __hash__(self):
        return hash(self.pos)


class AstarSearch:
    def __init__(
        self,
        Dr: DrGridRoute,
        CirDB: CustomSchematic,
        CurrentNet: RouteState,
        Drc: DrcManager,
        config: dict = None,
        net_config: dict = None,
        bStrictDRC: bool = True,
        groups: dict = None,
    ):
        self._Dr = Dr
        self._route_geometry = Dr.route_geometry
        self._cirdb = CirDB
        self._net = CurrentNet
        self._drcmgr = Drc
        self._config = config
        self._net_config = net_config
        self._bStrictDRC = bStrictDRC
        # self.taperLength = 10
        self.connect_th = math.ceil(
            config.dr.bend_radius * 2 / config.dr.grid_resolution
        )
        self._crossing_nets = set()
        self.resolution = config.dr.grid_resolution
        self.repel = False
        if self._config.dr.grid_resolution < 20:  # To be continued:
            self.repel = True
        self.groups = groups
        self.unrouted = max(2, min(10, Dr.unrouted))  # why 10?
        self.check_region = self.unrouted * config.dr.grid_resolution

        self.vAllNodesMap = {}

        self.penalty = {}

        self.penalty["propagation_loss"] = self._config.dr.loss_propagation
        self.penalty["bending_loss"] = self._config.dr.loss_bending
        self.penalty["crossing_loss"] = self._config.dr.loss_crossing
        self.penalty["congestion"] = self._config.dr.loss_congestion

        resolution = self._config.dr.grid_resolution
        if net_config:
            self.routing_bound = [
                self._drcmgr.bitMap.to_grid(net_config["region"][0]),
                self._drcmgr.bitMap.to_grid(net_config["region"][1]),
            ]
        else:
            scale_factor = (
                config.dr.net_bound_scale_factor + (self._net.failed_count // 2) * 0.5
            )
            x1, y1 = self._net.port1.center
            x2, y2 = self._net.port2.center
            x_mean = (x1 + x2) / 2
            y_mean = (y1 + y2) / 2

            boundx = (
                max(
                    abs(x1 - x2) * scale_factor,
                    resolution * config.dr.net_default_bound,
                )
                / 2
            )
            boundy = (
                max(
                    abs(y1 - y2) * scale_factor,
                    resolution * config.dr.net_default_bound,
                )
                / 2
            )
            lower = self._drcmgr.bitMap.to_grid([x_mean - boundx, y_mean - boundy])
            upper = self._drcmgr.bitMap.to_grid([x_mean + boundx, y_mean + boundy])
            self.routing_bound = [
                [max(1, int(lower[0])), max(1, int(lower[1]))],
                [
                    min(self._drcmgr.bitMap.width - 1, int(upper[0])),
                    min(self._drcmgr.bitMap.height - 1, int(upper[1])),
                ],
            ]

        CurrentNet.current_budget = CurrentNet.crossing_budget
        self.crossing_budget = CurrentNet.crossing_budget

        self.nodepq = []
        self._nodepq_counter = count()

        if net_config and not net_config["customH"]:
            self.costH = self.scaledMDist
        elif config.dr.enable_45_neighbor:
            # self.costH = self.customH
            self.costH = (
                lambda x, y, bending_loss=self.penalty["bending_loss"], res=self.resolution: (
                    customH(x, y, bending_loss, resolution=res)
                )
            )
        else:
            self.costH = self.scaledMDist

        self.stepG = {
            "straight_0": self.resolution * self.penalty["propagation_loss"],
            "straight_45": np.sqrt(2)
            * self.resolution
            * self.penalty["propagation_loss"],
        }

        self.init_state_machine()

    def init_state_machine(self):
        reslution = self._config.dr.grid_resolution
        radius = self._config.dr.bend_radius
        self.radius = radius
        self.grid_radius = math.ceil(radius / reslution)

        bend45_1 = math.ceil(radius * 0.415 / reslution)
        bend45_2 = math.ceil(radius * 0.3 / reslution)
        self.bend45_1 = bend45_1
        self.bend45_2 = bend45_2
        predict_length = round(self._net.half_size / reslution)
        self.set_DrcManager(self.grid_radius, bend45_1, bend45_2, predict_length)

        self.nextSteps = _build_transitions(
            self.grid_radius,
            bend45_1,
            bend45_2,
            enable_45=self._net.enable_45 and self._config.dr.enable_45_neighbor,
        )

        self.his_index = {0: 0, 45: 1, 90: 2, 135: 3, 180: 4, 225: 5, 270: 6, 315: 7}

        points = np.array(
            [
                (0, 0),
                (bend45_1 * reslution, 0),
                ((bend45_1 + bend45_2) * reslution, bend45_2 * reslution),
            ]
        )
        P = gf.path.smooth(
            points=points,
            radius=radius - 1e-3,
            bend=gf.path.arc,  # gf.path.euler / gf.path.arc
            # use_eff=True
        )
        length_45_1 = P.length()
        self.stepG["bend_45_1"] = (
            length_45_1 * self.penalty["propagation_loss"]
            + self.penalty["bending_loss"] / 2
        )

        self.stepG["bend_45_2"] = self.stepG["bend_45_1"]

        points = np.array(
            [
                (0, 0),
                (self.grid_radius * reslution, 0),
                (self.grid_radius * reslution, self.grid_radius * reslution),
            ]
        )
        P = gf.path.smooth(
            points=points,
            radius=radius - 1e-3,
            bend=gf.path.arc,  # gf.path.euler / gf.path.arc
            # use_eff=True
        )
        self.stepG["bend_90"] = (
            P.length() * self.penalty["propagation_loss"] + self.penalty["bending_loss"]
        )

    def set_DrcManager(self, bend90, bend45_1, bend45_2, predict_length):
        self._drcmgr.ben_45_part1 = bend45_1
        self._drcmgr.ben_45_part2 = bend45_2
        self._drcmgr.predict_length = predict_length
        self._drcmgr.radius = bend90

    def findAcsPts(self):
        """To do: only support positive index now"""
        if self._config.dr.group:
            if self._net.reverse:
                self.start_port = self._net.NetPort2
                self.end_port = self._net.NetPort1
            else:
                self.end_port = self._net.NetPort2
                self.start_port = self._net.NetPort1

            if self._net.earlyaccess:  ### To be continued
                access_grid = int(self.radius / self.resolution)
                endPos = self.end_port.port_grids[-access_grid]
            else:
                endPos = self.end_port.port_grids[0]
        else:
            self.start_port = self._net.NetPort1
            self.end_port = self._net.NetPort2
            endPos = self.end_port.port_grids[0]

        self.access_points = set(self.end_port.port_grids)

        self.endNode = GridAstarNode(
            pos=(endPos[0], endPos[1], self.end_port.orientation), crossing_budget=0
        )

        """To do: find start point considering the taper length
        """
        self.startNode = []

        startPos = self.start_port.port_grids[0]
        self.startNode.append(
            GridAstarNode(
                pos=(startPos[0], startPos[1], self.start_port.orientation),
                cost_g=0,
                cost_f=self.costH(startPos, endPos),
                straight_count=0,
                neighbors={},
                crossing_budget=self.crossing_budget,
            )
        )
        return True

    def connection_check(self, port1, port2):
        center1 = port1.center
        ori1 = round(port1.orientation)
        if ori1 == 360:
            ori1 = 0
        center2 = port2.center
        ori2 = round(port2.orientation)
        if ori2 == 360:
            ori2 = 0
        angle = abs(ori1 - ori2)
        if angle > 180:
            angle -= 180

        dx = abs(center1[0] - center2[0])
        dy = abs(center1[1] - center2[1])

        """ To do: Parametric coding
            """
        # gdsfactory do not support 180 bending
        if angle == 0:
            if ori1 == 0 or ori1 == 180:  ### lack 45 / 135 case
                if dy > 2.5 * self.radius:
                    return True, "all_angle", 180
                else:
                    return False, "None", 0
            elif dx > 2.5 * self.radius:
                return True, "all_angle", 180

        elif angle == 180:
            if ori1 == 90 or ori1 == 270:
                if dy > np.sqrt(dx * (4 * self.radius - dx)):
                    return True, "Sbend_y", 0
                else:
                    return False, "None", 0
            if ori1 == 0 or ori1 == 180:
                if dx > np.sqrt(dy * (4 * self.radius - dy)):
                    return True, "Sbend_x", 0
                else:
                    return False, "None", 0
        elif angle == 90:
            if dx >= self.radius and dy >= self.radius:
                ori1_rad = np.deg2rad(ori1)
                ori2_rad = np.deg2rad(ori2)
                dx1 = 100 * np.cos(ori1_rad)
                dy1 = 100 * np.sin(ori1_rad)
                dx2 = 100 * np.cos(ori2_rad)
                dy2 = 100 * np.sin(ori2_rad)
                center1_far = np.asarray(center1) + [dx1, dy1]
                l1 = sg.LineString([center1, center1_far])

                center2_far = np.asarray(center2) + [dx2, dy2]
                l2 = sg.LineString([center2, center2_far])

                intersect = l1.intersection(l2)
                if isinstance(intersect, sg.Point):
                    return True, "all_angle", 90
                return False, "None", 0
            else:
                return False, "None", 0

        elif angle == 45 or angle == 135:
            if ori1 == 45 or ori1 == 135 or ori1 == 225 or ori1 == 315:
                if ori2 == 0 or ori2 == 180:
                    if dy >= 0.3 * self.radius and dx >= dy + 0.4 * self.radius:
                        ori1_rad = np.deg2rad(ori1)
                        ori2_rad = np.deg2rad(ori2)
                        dx1 = 100 * np.cos(ori1_rad)
                        dy1 = 100 * np.sin(ori1_rad)
                        dx2 = 100 * np.cos(ori2_rad)
                        dy2 = 100 * np.sin(ori2_rad)
                        center1_far = np.asarray(center1) + [dx1, dy1]
                        l1 = sg.LineString([center1, center1_far])

                        center2_far = np.asarray(center2) + [dx2, dy2]
                        l2 = sg.LineString([center2, center2_far])

                        intersect = l1.intersection(l2)
                        if isinstance(intersect, sg.Point):
                            return True, "all_angle", 45

                        return False, "None", 0

                elif ori2 == 90 or ori2 == 270:
                    if dx >= 0.3 * self.radius and dy >= dx + 0.4 * self.radius:
                        ori1_rad = np.deg2rad(ori1)
                        ori2_rad = np.deg2rad(ori2)
                        dx1 = 100 * np.cos(ori1_rad)
                        dy1 = 100 * np.sin(ori1_rad)
                        dx2 = 100 * np.cos(ori2_rad)
                        dy2 = 100 * np.sin(ori2_rad)
                        center1_far = np.asarray(center1) + [dx1, dy1]
                        l1 = sg.LineString([center1, center1_far])

                        center2_far = np.asarray(center2) + [dx2, dy2]
                        l2 = sg.LineString([center2, center2_far])

                        intersect = l1.intersection(l2)
                        if isinstance(intersect, sg.Point):
                            return True, "all_angle", 45

                        return False, "None", 0

                return False, "None", 0
        else:
            return False, "None", 0

    def hard_connection(self, currentNode, radius=5):
        path, origin_path, violatedNets = self.backTrack(currentNode)
        """To be continued
        """
        if path is None:
            for node in origin_path[1:]:
                self.vAllNodesMap[node].visited = False
            return False

        Plength = len(path)
        # bend_circular has some problem...
        bend_circular = partial(gf.components.bend_circular, radius=radius)
        bend_euler = partial(gf.components.bend_euler, radius=radius)
        bend = bend_euler
        for _ in range(self.radius):
            try:
                segments, accumulated_bend = prepare_waypoints(path[0:Plength])
                P = gf.path.smooth(
                    points=segments,
                    radius=radius - 1e-3,
                    bend=gf.path.euler,
                    # bend=gf.path.arc,
                    use_eff=True,
                )
            except (TypeError, ValueError):
                Plength -= 1
                continue

            """can change the orther
            """
            # only support 0.5 width cross section now
            route = self._route_geometry.extrude(P)

            rect_route = self.fast_split_polygon_to_rectangles(
                route, self._drcmgr.resolution
            )
            # save result into net and update the bitmap
            if self.resolution == 1:
                expanded_route = self._route_geometry.extrude(P, width=2)
                expanded_rect_route = self.fast_split_polygon_to_rectangles(
                    expanded_route, self._drcmgr.resolution
                )
                rect_union = self.sets_union(expanded_rect_route, rect_route)
                self._drcmgr.updateBitmap(rect_route, self._net.netName)
            else:
                pass

            """Native ports are used for access-waveguide generation.
            """
            if self._bStrictDRC:
                # align = alignment(segments, self.start_port.port, self.end_port.port, radius=radius-1e-3, bend=gf.path.euler)
                self._net.rect_route = rect_route
                # if align is not None:
                # segments = align
                self._net.wg_component.append(route)
                # self._net.wg_component.append(temp1.flatten())
                # self._net.wg_component.append(temp2.flatten())
                self._net.routed_path = [
                    [segments.tolist(), self.start_port.port, self.end_port.port]
                ]
                self._net.crossing_nets = self._crossing_nets
                # self._net.current_budget -= len(self._crossing_nets)
                self._net.bending = accumulated_bend  # + connect_angle
                self._net.wirelength = P.length()
                self._net.crossing_num = len(self._crossing_nets)
            else:
                self._net.rect_route = rect_route  # save for updating bitmap
                self._net.origin_path = origin_path  # save for updating historyCost
                # self._drcmgr.updateBitmap(rect_route, self._net.netName)
                self._net.wg_component.append(route)
                # self._net.wg_component.append(temp1.flatten())
                # self._net.wg_component.append(temp2.flatten())
                self._net.routed_path = [
                    [segments.tolist(), self.start_port.port, self.end_port.port]
                ]
                self._net.wirelength = P.length()
                self._net.bending = accumulated_bend
                self._Dr.mark_conflict_nets(violatedNets, self._net.netName)
                self._net.vioNets = violatedNets
                self._net.vionets = len(violatedNets)
            return True

    def sets_union(self, sets1, sets2):
        curve_rects_1, straight_90_1, straight_0_1, straight_45_1, straight_135_1 = (
            sets1
        )
        curve_rects_2, straight_90_2, straight_0_2, straight_45_2, straight_135_2 = (
            sets2
        )
        curve_rects = curve_rects_1.union(curve_rects_2)
        straight_90 = straight_90_1.union(straight_90_2)
        straight_0 = straight_0_1.union(straight_0_2)
        straight_45 = straight_45_1.union(straight_45_2)
        straight_135 = straight_135_1.union(straight_135_2)
        return (curve_rects, straight_90, straight_0, straight_45, straight_135)

    def route(self):
        if (
            self._net.distance["Euler"] < 10
            and abs(self.startNode[0].pos[2] - self.endNode.pos[2]) == 180
        ):
            start_port = self.start_port.port
            access_waveguide, access_route = (
                self._route_geometry.connect_ports_all_angle(
                    start_port, self.end_port.port
                )
            )
            self._net.wirelength += self._route_geometry.route_length_um(
                access_waveguide, access_route
            )
            self._net.wg_component.append(access_waveguide)
            self._cirdb.layout.add_ref(access_waveguide)
            return True

        if not self.findAcsPts():
            return False
        for node in self.startNode:
            self._push_node(node)

        while (current_node := self._pop_node()) is not None:
            current_node.visited = True

            if self.check_connection(
                current_node.pos, self.endNode.pos
            ):  # self.connect_th:
                if self.hard_connection(
                    current_node, radius=self._config.dr.bend_radius
                ):
                    self._net.routed = True
                    return True
                else:
                    current_node.visited = False

            # find the current node's neightbors and loop through the neighbors
            if not current_node.neighbors:
                self.findNeighbors(current_node)

            for neighbor, nb_type in current_node.neighbors.items():
                if neighbor.visited:
                    continue
                costF = 0
                if nb_type in {"crossing_0", "crossing_45"}:
                    costG = current_node.cost_g
                    costG += self.penalty["crossing_loss"]
                else:
                    costG = current_node.cost_g + self.stepG[nb_type]

                if neighbor.is_violated:
                    cv = current_node.violated_nets
                    nv = neighbor.violated_nets
                    union_nets = cv | nv
                    netnum = len(union_nets)
                    for net in nv:
                        if net in cv:
                            costG += 0.1 * self.penalty["crossing_loss"]
                        else:
                            costG += netnum * 10 * self.penalty["crossing_loss"]

                x, y, ori = neighbor.pos

                if self.groups:
                    dis = EulerDist(neighbor.pos, self.endNode.pos) * self.resolution
                    # congestion_penalty = self.penalty["bending_loss"] * 0.1
                    congestion_penalty = self.penalty["congestion"]
                    if dis >= self.check_region:  # compatible with NOC and computing
                        if (
                            self.repel
                            and (ori + self.end_port.orientation) % 180 != 0
                            and self._net.failed_count > 1
                        ):  #
                            count = 0
                            if nb_type == "straight_0":
                                count = self._drcmgr.checkSpacing(neighbor, 5, {})
                                costG += count * congestion_penalty
                            elif nb_type == "straight_45":
                                count = self._drcmgr.checkSpacing(neighbor, 4, {})
                                costG += count * congestion_penalty

                        if not self.repel:
                            count = 0
                            if nb_type == "straight_0":
                                count = self._drcmgr.checkSpacing(
                                    neighbor, self.unrouted, self.groups
                                )
                            elif nb_type == "straight_45":
                                count = self._drcmgr.checkSpacing(
                                    neighbor, self.unrouted, self.groups
                                )
                            if count > 0:
                                costG += count * congestion_penalty

                if self.bNeedUpdate(neighbor, costG):
                    costF += costG + self.penalty["propagation_loss"] * self.costH(
                        neighbor.pos, self.endNode.pos
                    )

                    if self._bStrictDRC:
                        his = self._Dr.historyMap[x][y][self.his_index[ori]]
                        if his:
                            costF += his

                    if nb_type in {"straight_0", "straight_45"}:
                        neighbor.straight_count = current_node.straight_count + 1
                    elif nb_type in {"bend_45_1", "bend_45_2"}:
                        neighbor.straight_count = 0

                    neighbor.cost_g = costG
                    neighbor.cost_f = costF
                    neighbor.parent = current_node
                    self._push_node(neighbor)

        return False

    def _push_node(self, node: GridAstarNode) -> None:
        """Add a node snapshot, preserving insertion order for equal costs."""
        heappush(
            self.nodepq,
            (node.cost_f, next(self._nodepq_counter), node),
        )

    def _pop_node(self) -> Optional[GridAstarNode]:
        """Return the cheapest current node, discarding stale heap entries."""
        while self.nodepq:
            priority, _, node = heappop(self.nodepq)
            if not node.visited and priority == node.cost_f:
                return node
        return None

    def findNeighbors(self, currentNode: GridAstarNode):
        currentX, currentY, orientation = currentNode.pos
        nextStep = self.nextSteps[orientation]
        end_node = self.endNode.pos
        th = self.connect_th
        for step in nextStep:
            enable_prediction = True
            neighbor_position = (currentX + step[0], currentY + step[1], step[2])
            dis = manhattanDH(currentNode.pos, end_node)
            if dis <= th:
                enable_prediction = False
            routing_bound = self.routing_bound
            if (
                neighbor_position[0] <= routing_bound[0][0]
                or neighbor_position[0] >= routing_bound[1][0]
                or neighbor_position[1] <= routing_bound[0][1]
                or neighbor_position[1] >= routing_bound[1][1]
            ):
                continue
            result = self._drcmgr.bViolateDRC(
                currentNode,
                step,
                self._bStrictDRC,
                self._net.netName,
                enable_prediction,
            )
            vAllNodesMap = self.vAllNodesMap
            if not result.violated:
                tmp_neighbor = vAllNodesMap.get(neighbor_position, None)
                if tmp_neighbor is None:
                    tmp_neighbor = vAllNodesMap[neighbor_position] = GridAstarNode(
                        pos=neighbor_position,
                        parent=None,
                        neighbors={},
                        crossing_budget=currentNode.crossing_budget,
                    )
                currentNode.add_neighbor(tmp_neighbor, step[3])

            elif result.crossing_position is not None:
                tmp_neighbor = vAllNodesMap.get(result.crossing_position, None)
                if tmp_neighbor is None:
                    tmp_neighbor = vAllNodesMap[result.crossing_position] = (
                        GridAstarNode(
                            pos=result.crossing_position,
                            parent=None,
                            neighbors={},
                            crossing_budget=currentNode.crossing_budget - 1,
                            crossing_net=result.conflicts,
                            # is_violated=ans[4]
                        )
                    )
                currentNode.add_neighbor(tmp_neighbor, result.obstacle_type)

            elif not self._bStrictDRC and result.obstacle_type not in {"blk", "port"}:
                tmp_neighbor = vAllNodesMap.get(neighbor_position, None)
                if tmp_neighbor is None:
                    tmp_neighbor = vAllNodesMap[neighbor_position] = GridAstarNode(
                        pos=neighbor_position,
                        parent=None,
                        neighbors={},
                        is_violated=True,
                        violated_nets=result.conflicts,
                    )
                currentNode.add_neighbor(tmp_neighbor, step[3])

    def scaledMDist(self, pos1, pos2):
        x = abs(pos1[0] - pos2[0]) * self.resolution
        y = abs(pos1[1] - pos2[1]) * self.resolution
        if x > 0 and y > 0:
            return x + y + self.resolution
        else:
            return x + y

    def check_connection(self, pos1, pos2):
        x1, y1, ori1 = pos1
        x2, y2, ori2 = pos2
        if (x1, y1) in self.access_points and abs(ori1 - ori2) == 180:
            return True
        else:
            return False

    def backTrack(self, node: GridAstarNode):
        if not self._bStrictDRC:
            violatedNets = set()
            path = [node.pos[0:]]
            if node.violated_nets:
                violatedNets = violatedNets | node.violated_nets
            while node.parent is not None:
                path.append(node.parent.pos[0:])
                if node.violated_nets:
                    violatedNets = violatedNets | node.violated_nets
                node = node.parent
            path = path[::-1]
            origin_path = deepcopy(path)
            path = np.array(self._process_bend(path))
            path[:, 0:2] = self._drcmgr.bitMap.to_world(path[:, 0:2])
            return path[:, 0:2], origin_path, violatedNets
        else:
            flag = False  # used to check whether the net is crossed twice
            path = [node.pos[0:]]
            self._crossing_nets.clear()
            if node.crossing_net:
                self._crossing_nets.add(node.crossing_net)
            while node.parent is not None:
                path.append(node.parent.pos[0:])
                node = node.parent
                if node.crossing_net:
                    if node.crossing_net in self._crossing_nets:
                        flag = True
                    else:
                        self._crossing_nets.add(node.crossing_net)
            path = path[::-1]
            origin_path = deepcopy(path)
            if flag:
                return None, origin_path, None
            path = np.array(self._process_bend(path))
            path[:, 0:2] = self._drcmgr.bitMap.to_world(path[:, 0:2])
            return path[:, 0:2], origin_path, None

    def _process_bend(self, path):
        vec = []
        if len(path) == 1:
            vec.append(path[0][0:])
            return vec
        if len(path) == 2:
            vec.append(path[0][0:])
            vec.append(path[1][0:])
            return vec

        k = len(path)
        for i in range(0, k):
            vec.append(path[i][0:])
            if i == k - 1:
                break
            elif path[i][2] != path[i + 1][2]:  # next point
                ori = path[i][2]
                angle = abs(path[i + 1][2] - ori)
                if angle > 180:
                    angle = 360 - angle
                if angle == 45:
                    match ori:
                        case 0:
                            vec.append((path[i][0] + self.bend45_1, path[i][1], ori))
                        case 90:
                            vec.append((path[i][0], path[i][1] + self.bend45_1, ori))
                        case 180:
                            vec.append((path[i][0] - self.bend45_1, path[i][1], ori))
                        case 270:
                            vec.append((path[i][0], path[i][1] - self.bend45_1, ori))
                        case 45:
                            vec.append(
                                (
                                    path[i][0] + self.bend45_2,
                                    path[i][1] + self.bend45_2,
                                    ori,
                                )
                            )
                        case 135:
                            vec.append(
                                (
                                    path[i][0] - self.bend45_2,
                                    path[i][1] + self.bend45_2,
                                    ori,
                                )
                            )
                        case 225:
                            vec.append(
                                (
                                    path[i][0] - self.bend45_2,
                                    path[i][1] - self.bend45_2,
                                    ori,
                                )
                            )
                        case 315:
                            vec.append(
                                (
                                    path[i][0] + self.bend45_2,
                                    path[i][1] - self.bend45_2,
                                    ori,
                                )
                            )

                else:
                    match ori:
                        case 0:
                            vec.append((path[i][0] + self.grid_radius, path[i][1], ori))
                        case 90:
                            vec.append((path[i][0], path[i][1] + self.grid_radius, ori))
                        case 180:
                            vec.append((path[i][0] - self.grid_radius, path[i][1], ori))
                        case 270:
                            vec.append((path[i][0], path[i][1] - self.grid_radius, ori))
        return vec

    def _mergePath(self, path: List[Tuple[int]]) -> List[Tuple[int]]:
        vec = []
        if len(path) == 1:
            vec.append(path[0][0:2])
            return vec
        if len(path) == 2:
            vec.append(path[0][0:2])
            vec.append(path[1][0:2])
            return vec

        i = 0
        k = len(path)
        for j in range(1, k + 1):
            if j != k:
                if i == j:
                    continue
                v1 = path[j]
                # same direction
                if path[i][2] == v1[2]:
                    continue

                angle = abs(v1[2] - path[i][2])
                if angle > 180:
                    angle = 360 - angle
                if angle == 45:
                    vec.append(path[i][0:2])
                    # more than two node in the same direction, add the second point of the segment
                    if i != j - 1:
                        vec.append(path[j - 1][0:2])
                        i = j
                    else:
                        i = j
                # angle == 90, add the inflection node, ignore the intermediate node
                elif path[i][2] == 180:
                    vec.append(path[i][0:2])
                    vec.append((path[j - 1][0] - 5, path[j - 1][1]))
                    i = j
                elif path[i][2] == 0:
                    vec.append(path[i][0:2])
                    vec.append((path[j - 1][0] + 5, path[j - 1][1]))
                    i = j
                elif path[i][2] == 90:
                    vec.append(path[i][0:2])
                    vec.append((path[j - 1][0], path[j - 1][1] + 5))
                    i = j
                else:
                    vec.append(path[i][0:2])
                    vec.append((path[j - 1][0], path[j - 1][1] - 5))
                    i = j
            else:  # The last node
                if i == k - 1:
                    vec.append(path[i][0:2])
                else:
                    vec.append(path[i][0:2])
                    vec.append(path[k - 1][0:2])

        return vec

    def hasBend(self, currentNode: GridAstarNode, neighbor: GridAstarNode):
        return currentNode.pos[2] != neighbor.pos[2]

    def _findDirection(self, currentNode: GridAstarNode, neighbor: GridAstarNode):
        if currentNode.pos[0] == neighbor.pos[0]:
            assert currentNode.pos[1] != neighbor.pos[1]
            return "n" if currentNode.pos[1] < neighbor.pos[1] else "s"
        elif currentNode.pos[1] == neighbor.pos[1]:
            return "e" if currentNode.pos[0] < neighbor.pos[0] else "w"
        elif currentNode.pos[0] < neighbor.pos[0]:
            return "ne" if currentNode.pos[1] < neighbor.pos[1] else "se"
        else:
            return "nw" if currentNode.pos[1] < neighbor.pos[1] else "sw"

    def bNeedUpdate(self, neighbor: GridAstarNode, costG):
        if neighbor.cost_g > costG:
            return True
        return False

    def fast_split_polygon_to_rectangles(self, component: gf.Component, resolution=1):
        """Classify filled route cells using GDSFactory/KLayout geometry."""
        layer = gf.get_layer_tuple(self._route_geometry.layer)
        points = component.get_polygons_points(
            by="tuple", layers=[self._route_geometry.layer]
        )[layer][0]
        grid_points = (points - self._drcmgr.bitMap.origin) / self._drcmgr.resolution
        int_points = grid_points.astype(np.int32).tolist()
        int_points = tuple(map(tuple, int_points))
        dpoint = points[0:-1, :] - points[1:, :]
        slope = np.divide(
            dpoint[:, 1],
            dpoint[:, 0],
            out=np.copysign(np.full(len(dpoint), np.inf), dpoint[:, 1]),
            where=dpoint[:, 0] != 0,
        )
        grid_dict = {}
        for i, num in enumerate(dpoint):
            abs_num = abs(num)
            if abs_num[0] > 0.5 or abs_num[1] > 0.5:  # Interpolation
                num_points = round(max(abs_num[0], abs_num[1]) / 0.1)
                x = (
                    np.linspace(
                        grid_points[i, 0],
                        grid_points[i + 1, 0],
                        num_points,
                    )
                    .astype(np.int32)
                    .tolist()
                )
                y = (
                    np.linspace(
                        grid_points[i, 1],
                        grid_points[i + 1, 1],
                        num_points,
                    )
                    .astype(np.int32)
                    .tolist()
                )
                for grid in tuple(zip(x, y)):
                    if grid_dict.get(grid) is None:
                        grid_dict[grid] = []
                    grid_dict[grid].append(slope[i])

        curve_rects = set()
        straight_0 = set()
        straight_45 = set()
        straight_90 = set()
        straight_135 = set()

        for i, num in enumerate(slope):
            if grid_dict.get(int_points[i]) is None:
                grid_dict[int_points[i]] = []
            grid_dict[int_points[i]].append(slope[i])

        # Boundary slopes provide direction; the KLayout region supplies the
        # authoritative filled cells. Interior cells are conservatively bends.
        for grid in self._route_geometry.occupied_grid_cells(
            component, self._drcmgr.bitMap
        ):
            grid_dict.setdefault(grid, [np.nan])

        for grid, slope in grid_dict.items():
            with np.errstate(invalid="ignore"):
                mean_slope = np.mean(slope)
            abs_slope = np.abs(slope)
            abs_mean_slope = np.mean(abs_slope)
            min_slope = np.min(abs_slope)
            grid_dict[grid] = mean_slope
            # contain inf
            if (
                np.isnan(mean_slope)
                or mean_slope == float("inf")
                or mean_slope == float("-inf")
            ):  # [inf, -inf] / [inf, ...] / [-inf, ...]
                if min_slope == float("inf"):  # [-inf,..., inf]
                    straight_90.add((grid[0], grid[1]))
                    continue
                result = Counter(abs_slope)
                if len(result) == 2:
                    if result[0] > result[float("inf")]:
                        straight_0.add((grid[0], grid[1]))
                    else:
                        straight_90.add((grid[0], grid[1]))
                else:
                    curve_rects.add((grid[0], grid[1]))
            # 45/135
            elif abs(abs_mean_slope - 1) < 1e-2:
                if mean_slope > 0:
                    straight_45.add((grid[0], grid[1]))
                if mean_slope < 0:
                    straight_135.add((grid[0], grid[1]))
            # 0/180
            elif abs(mean_slope) < 1e-3:
                straight_0.add((grid[0], grid[1]))
            # 90/270
            elif abs(mean_slope) > 3e3:
                straight_90.add((grid[0], grid[1]))
            else:
                curve_rects.add((grid[0], grid[1]))

        return (curve_rects, straight_90, straight_0, straight_45, straight_135)
