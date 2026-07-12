import math
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..database.schematic import CustomSchematic
from .bitmap import Bitmap

_PERPENDICULAR_STEPS = {
    0: (0, 1),
    45: (1, -1),
    90: (1, 0),
    135: (1, 1),
    180: (0, 1),
    225: (1, -1),
    270: (1, 0),
    315: (1, 1),
}


@dataclass(frozen=True, slots=True)
class DrcResult:
    violated: bool
    crossing_position: tuple[int, int, int] | None = None
    obstacle_type: Any = None
    conflicts: set[str] | str | None = None


class DrcManager(object):
    def __init__(self, Cirdb: CustomSchematic, config):
        self._cirdb = Cirdb

        self.dieSize = Cirdb.schematic.settings.die_area  # Shape: [[],[]]
        self.resolution = config.dr.grid_resolution
        self._bitmap = Bitmap(self.dieSize, self.resolution)
        self.checkSteps = {
            0: (1, 0),
            45: (1, 1),
            90: (0, 1),
            135: (-1, 1),
            180: (-1, 0),
            225: (-1, -1),
            270: (0, -1),
            315: (1, -1),
        }

        # default
        self.radius = math.ceil(config.dr.bend_radius / self.resolution)
        self.ben_45_part1 = 3
        self.ben_45_part2 = 2
        self.predict_length = 3

        self.width = self._bitmap.width
        self.height = self._bitmap.height

        self.straight = 1
        self.maxCrossing = 10
        self.port_length = math.ceil(
            (self.maxCrossing + config.dr.bend_radius) / self.resolution
        )

    @property
    def bitMap(self):
        return self._bitmap

    @property
    def bitMap_width(self):
        return self._bitmap.width

    @property
    def bitMap_height(self):
        return self._bitmap.height

    """
    initialize the bitmap based on the input components' bounding box 
    """

    def initDRC(self):
        self.initBitmap()

    def initBitmap(self):
        self._bitmap.initMap(self._cirdb.initBlk.values())

    def initPorts(self):
        for ins_name, blk in self._cirdb.initBlk.items():
            spread_ports = self.spread_ports
            ref = self._cirdb._inst_blks[ins_name]

            ports_0 = ref.get_ports_list(orientation=0)
            ports_90 = ref.get_ports_list(orientation=90)
            ports_180 = ref.get_ports_list(orientation=180)
            ports_270 = ref.get_ports_list(orientation=270)
            spread_ports(ins_name, ports_0, ori=0)
            spread_ports(ins_name, ports_90, ori=90)
            spread_ports(ins_name, ports_180, ori=180)
            spread_ports(ins_name, ports_270, ori=270)

    def spread_ports(self, ins_name, ports, ori=0, spacing=12):
        orientations = {
            0.0: np.array([1, 0]),
            90.0: np.array([0, 1]),
            180.0: np.array([-1, 0]),
            270.0: np.array([0, -1]),
        }
        ports_num = len(ports)
        match ports_num:
            case 0:
                return
            case 1:
                Port = ports[0]
                inst_group = {}
                nets = {}
                name = ins_name + "_" + str(ori)

                if Port.netName:
                    net = self._cirdb.dbNets[Port.netName]
                    inst = self._cirdb._inst_blks[net.NetPort2.instanceName]
                    orient = net.NetPort2.orientation
                    port_len = len(inst.get_ports_list(orient))

                    if net.NetPort1 == Port and port_len <= 2:
                        net.reverse = True
                    nets[Port.netName] = None
                    net.groups.append(name)
                    inst_group["nets"] = nets
                    inst_group["configs"] = None
                    self._cirdb._group_Nets[name] = inst_group

                    step = orientations[Port.orientation]
                    loc = self._bitmap.to_grid(ports[0].center)
                    while self._bitmap.bitMap[loc[0]][loc[1]].type == "blk":
                        loc += step
                    for _ in range(self.port_length):
                        self._bitmap.bitMap[loc[0]][loc[1]].update("port", Port.netName)
                        Port.port_grids.append((loc[0], loc[1]))
                        loc += step
            case _:
                CR = 0
                x1, y1 = ports[0].center
                x2, y2 = ports[1].center
                space = abs(x1 - x2) + abs(y1 - y2)

                def spread(orientation):
                    if orientation == 0 or orientation == 180:
                        spread_axis, origin_axis = 1, 0
                    else:
                        spread_axis, origin_axis = 0, 1
                    order_ports = [
                        (p[0].port_name, p[0].center)
                        for p in sorted(  # based on orientation
                            [(port, port.center[spread_axis]) for port in ports],
                            key=lambda x: x[1],
                        )
                    ]
                    inst_group = {}
                    nets = {}
                    name = ins_name + "_" + str(ori)
                    step = orientations[ori]
                    mean_loc = self._bitmap.to_grid(
                        (np.array(order_ports[0][1]) + np.array(order_ports[-1][1])) / 2
                    )
                    while self._bitmap.bitMap[mean_loc[0]][mean_loc[1]].type == "blk":
                        mean_loc += step
                    if CR < 100000:
                        if self.resolution == 1:  # exist some problems
                            if space <= 2:  # spread
                                half_port = ports_num / 2
                                spread_axis_min = mean_loc[spread_axis] - half_port
                                spread_axis_max = spread_axis_min + (ports_num - 1) * 2
                                spread_points = np.linspace(
                                    spread_axis_min, spread_axis_max, ports_num
                                ).astype(np.int32)
                                for i, port in enumerate(order_ports):
                                    origin_mean = mean_loc[origin_axis]
                                    Port = self._cirdb._dbPorts.get(port[0], None)
                                    if Port.netName:
                                        if spread_axis:
                                            loc = (
                                                np.array(
                                                    [origin_mean, spread_points[i]]
                                                )
                                            ).astype(np.int32)
                                        else:
                                            loc = (
                                                np.array(
                                                    [spread_points[i], origin_mean]
                                                )
                                            ).astype(np.int32)
                                        for _ in range(self.port_length):
                                            self._bitmap.bitMap[loc[0]][loc[1]].update(
                                                "port", Port.netName
                                            )
                                            Port.port_grids.append((loc[0], loc[1]))
                                            loc += step
                            else:
                                half_port = ports_num / 2
                                for i, port in enumerate(order_ports):
                                    port_name = ins_name + port[0]
                                    Port = self._cirdb._dbPorts.get(port_name, None)
                                    loc = (np.array(port[1])).astype(np.int32)
                                    loc[0] = mean_loc[0]
                                    if i < half_port:
                                        port_len = (
                                            i + 1
                                        ) * self.port_length + self.radius
                                    else:
                                        port_len = (
                                            ports_num - i
                                        ) * self.port_length + self.radius
                                    net = self._cirdb.dbNets[Port.netName]
                                    net.earlyaccess = True
                                    if net.NetPort1 == Port:
                                        self._cirdb.dbNets[Port.netName].reverse = True
                                    for _ in range(port_len):
                                        self._bitmap.bitMap[loc[0]][loc[1]].update(
                                            "port", Port.netName
                                        )
                                        Port.port_grids.append((loc[0], loc[1]))
                                        loc += step
                        elif space > self.resolution:
                            half_port = ports_num / 2
                            for i, port in enumerate(order_ports):
                                Port = self._cirdb._dbPorts.get(port[0], None)
                                if Port.netName:
                                    nets[Port.netName] = None
                                    loc = self._bitmap.to_grid(port[1])
                                    loc[origin_axis] = mean_loc[origin_axis]
                                    if ports_num <= 2:
                                        port_len = self.radius * 4  ### To be continued
                                    else:
                                        if i < half_port - 1:
                                            port_len = (
                                                i * self.port_length + self.radius
                                            )  # normalized by resolution
                                        elif i == half_port - 1:
                                            port_len = (
                                                i + 1
                                            ) * self.port_length + self.radius
                                        else:
                                            port_len = (
                                                ports_num - i - 1
                                            ) * self.port_length + self.radius
                                    net = self._cirdb.dbNets[Port.netName]
                                    net.earlyaccess = True
                                    net.groups.append(name)  # add routing group
                                    if (
                                        net.NetPort1 == Port
                                        and "fanout" not in ins_name
                                    ):
                                        net.reverse = True
                                    for _ in range(port_len):
                                        self._bitmap.bitMap[loc[0]][loc[1]].update(
                                            "port", Port.netName
                                        )
                                        Port.port_grids.append((loc[0], loc[1]))
                                        loc += step
                            inst_group["nets"] = nets
                            inst_group["configs"] = None
                            self._cirdb._group_Nets[name] = inst_group
                        else:
                            half_port = ports_num / 2
                            spread_axis_min = int(mean_loc[spread_axis] - half_port)
                            spread_axis_max = int(mean_loc[spread_axis] + half_port)
                            sine_step = math.ceil(
                                5 / self.resolution
                            )  ### To be continued: 5
                            origin_mean = mean_loc[origin_axis]
                            origin_sbend = origin_mean + sine_step * step[0]
                            spread_points = np.linspace(
                                spread_axis_min, spread_axis_max, ports_num
                            ).astype(np.int32)
                            origin_min = min(origin_mean, origin_sbend)
                            origin_max = max(origin_mean, origin_sbend)
                            origin_sbend += step[0]
                            if spread_axis:  ## Block the sine bend region
                                for j in range(
                                    int(spread_axis_min), int(spread_axis_max) + 1
                                ):
                                    self._bitmap.bitMap[origin_min][j].update("blk", -1)
                                    self._bitmap.bitMap[origin_max][j].update("blk", -1)
                                for i in range(int(origin_min), int(origin_max)):
                                    self._bitmap.bitMap[i][spread_axis_min].update(
                                        "blk", -1
                                    )
                                    self._bitmap.bitMap[i][spread_axis_max].update(
                                        "blk", -1
                                    )
                            else:
                                for j in range(
                                    int(spread_axis_min), int(spread_axis_max) + 1
                                ):
                                    self._bitmap.bitMap[j][origin_min].update("blk", -1)
                                    self._bitmap.bitMap[j][origin_max].update("blk", -1)
                                for i in range(int(origin_min), int(origin_max)):
                                    self._bitmap.bitMap[spread_axis_min][i].update(
                                        "blk", -1
                                    )
                                    self._bitmap.bitMap[spread_axis_max][i].update(
                                        "blk", -1
                                    )

                            for i, port in enumerate(order_ports):
                                Port = self._cirdb._dbPorts.get(port[0], None)
                                if Port.netName:
                                    net = self._cirdb.dbNets[Port.netName]
                                    nets[Port.netName] = None
                                    net.groups.append(name)
                                    # net.earlyaccess = True
                                    if (
                                        net.NetPort1 == Port
                                        and "fanout" not in ins_name
                                    ):  # Do not reverse the fanout routing
                                        net.reverse = True
                                    if spread_axis:
                                        loc = (
                                            np.array([origin_sbend, spread_points[i]])
                                        ).astype(np.int32)
                                    else:
                                        loc = (
                                            np.array([spread_points[i], origin_sbend])
                                        ).astype(np.int32)
                                    for _ in range(self.port_length):
                                        self._bitmap.bitMap[loc[0]][loc[1]].update(
                                            "port", Port.netName
                                        )
                                        Port.port_grids.append((loc[0], loc[1]))
                                        loc += step
                            inst_group["nets"] = nets
                            inst_group["configs"] = None
                            self._cirdb._group_Nets[name] = inst_group

                spread(ori)

    def updateBitmap(self, rectangles, netID):
        (curve_rects, straight_90, straight_0, straight_45, straight_135) = rectangles
        for rect in curve_rects:
            x, y = int(rect[0]), int(rect[1])
            self._cirdb.dbNets[netID].rwguide.add((x, y))
            self._bitmap.bitMap[x][y].update("waveguide", id=netID, wgtype=1)  # bend
        for rect in straight_90:
            x, y = int(rect[0]), int(rect[1])
            self._cirdb.dbNets[netID].rwguide.add((x, y))
            self._bitmap.bitMap[x][y].update("waveguide", id=netID, wgtype=90)
        for rect in straight_0:
            x, y = int(rect[0]), int(rect[1])
            self._cirdb.dbNets[netID].rwguide.add((x, y))
            self._bitmap.bitMap[x][y].update("waveguide", id=netID, wgtype=180)
        for rect in straight_45:
            x, y = int(rect[0]), int(rect[1])
            self._cirdb.dbNets[netID].rwguide.add((x, y))
            self._bitmap.bitMap[x][y].update("waveguide", id=netID, wgtype=45)
        for rect in straight_135:
            x, y = int(rect[0]), int(rect[1])
            self._cirdb.dbNets[netID].rwguide.add((x, y))
            self._bitmap.bitMap[x][y].update("waveguide", id=netID, wgtype=135)

    """To do:
    DRC check based on the crossing size
    """

    def _scan_route_segments(self, start, segments, net_name):
        """Scan consecutive route segments and collect conflicting route owners."""
        x, y = start
        conflicts = set()
        for (dx, dy), length, orientation in segments:
            for _ in range(length):
                x += dx
                y += dy
                violated, node_type, owner = self.checkSingleNode(
                    (x, y, orientation), net_name
                )
                if not violated or (node_type == "port" and owner == net_name):
                    continue
                if node_type in {"port", "blk"}:
                    return (
                        (x, y),
                        conflicts,
                        DrcResult(True, obstacle_type=node_type, conflicts=owner),
                    )
                conflicts.add(owner)
        return (x, y), conflicts, None

    def _prediction_is_blocked(self, start, step, orientation, net_name) -> bool:
        x, y = start
        try:
            for _ in range(self.predict_length):
                x += step[0]
                y += step[1]
                violated, node_type, owner = self.checkSingleNode(
                    (x, y, orientation), net_name
                )
                if violated and not (node_type == "port" and owner == net_name):
                    return True
        except IndexError:
            return True
        return False

    def bViolateDRC(
        self, currentNode, step, crossing_enable, netName, enable_prediction
    ):  # crossing enable
        currentNode_pos = currentNode.pos
        crossing_budget = currentNode.crossing_budget
        straight_count = currentNode.straight_count
        check_ori = step[2]  # next step orientation
        checkstep = self.checkSteps[check_ori]
        checkX = currentNode_pos[0]
        checkY = currentNode_pos[1]
        nb_type = step[3]  # check type
        dbNets = self._cirdb.dbNets

        checkcrossing = self.checkSingleNode
        if self.resolution > 1:
            checkBitmap = self.checkSingleNode
        elif enable_prediction:
            checkBitmap = self.checkSingleNode
        else:
            checkBitmap = self.checkSingleNode

        match nb_type:
            case "straight_0":
                checkX += checkstep[0]
                checkY += checkstep[1]
                bviolated, nType, nNet = checkBitmap(
                    (checkX, checkY, check_ori), netName
                )

                if bviolated:  # check crossing: return [bool, bitmapNode]
                    violated_net = {nNet}
                    # violated_net.add(nNet)
                    if nType == "port" and nNet == netName:
                        return DrcResult(False)
                    elif nType in {"blk", "compound", "port"}:
                        return DrcResult(
                            True, obstacle_type=nType, conflicts=violated_net
                        )
                    elif abs(check_ori - nType) == 90:
                        if (
                            not crossing_enable
                            or crossing_budget <= 0
                            or dbNets[nNet].current_budget <= 0
                        ):
                            return DrcResult(
                                True, obstacle_type=nType, conflicts=violated_net
                            )
                        # 90+180 / 270+180 / 0+90 / 180+90
                        host = dbNets[netName]
                        slave = dbNets[nNet]
                        crossing_neighbor = None
                        bcrossing, half_size = self.crossing_check(
                            host, slave, straight_count + 1, True
                        )
                        if bcrossing:
                            checking_size = round(half_size / self.resolution + 0.1)
                            if check_ori == 0 or check_ori == 180:  # check crossing
                                for i in range(1, checking_size + 1):
                                    if (
                                        checkcrossing((checkX, checkY + i), netName)[1]
                                        != nType
                                        or checkcrossing((checkX, checkY - i), netName)[
                                            1
                                        ]
                                        != nType
                                    ):
                                        return DrcResult(
                                            True,
                                            obstacle_type=nType,
                                            conflicts=violated_net,
                                        )
                            else:
                                for i in range(1, checking_size + 1):
                                    if (
                                        checkcrossing((checkX + i, checkY), netName)[1]
                                        != nType
                                        or checkcrossing((checkX - i, checkY), netName)[
                                            1
                                        ]
                                        != nType
                                    ):
                                        return DrcResult(
                                            True,
                                            obstacle_type=nType,
                                            conflicts=violated_net,
                                        )

                            crossing_neighbor = (checkX, checkY, check_ori)
                            for _ in range(checking_size):
                                checkX += checkstep[0]
                                checkY += checkstep[1]
                                crossing_neighbor = (checkX, checkY, check_ori)
                                cr_bviolated, cr_nType, cr_nNet = checkcrossing(
                                    crossing_neighbor, netName
                                )
                                if cr_bviolated:
                                    if cr_nType == "port" and cr_nNet == netName:
                                        continue
                                    else:
                                        return DrcResult(
                                            True,
                                            obstacle_type=nType,
                                            conflicts=violated_net,
                                        )

                            return DrcResult(
                                True, crossing_neighbor, "crossing_0", nNet
                            )
                        else:
                            return DrcResult(
                                True, obstacle_type=nType, conflicts=violated_net
                            )
                    return DrcResult(True, obstacle_type=nType, conflicts=violated_net)

            case "straight_45":
                checkX += checkstep[0]
                checkY += checkstep[1]
                bviolated, nType, nNet = checkBitmap(
                    (checkX, checkY, check_ori), netName
                )

                if bviolated:  # check crossing: return [bool, bitmapNode]
                    violated_net = set()
                    violated_net.add(nNet)
                    if nType == "port" and nNet == netName:
                        return DrcResult(False)
                    if nType in {"blk", "compound", "port"}:
                        return DrcResult(
                            True, obstacle_type=nType, conflicts=violated_net
                        )
                    if (
                        check_ori + nType
                    ) % 180 == 0:  # 45+135 / 225+135 / 135+45 / 315+45
                        if (
                            not crossing_enable
                            or crossing_budget <= 0
                            or dbNets[nNet].current_budget <= 0
                        ):
                            return DrcResult(
                                True, obstacle_type=nType, conflicts=violated_net
                            )
                        host = dbNets[netName]
                        slave = dbNets[nNet]
                        crossing_neighbor = None
                        bcrossing, half_size = self.crossing_check(
                            host, slave, straight_count + 1, False
                        )
                        if bcrossing:
                            checking_size = round(half_size / (self.resolution) + 0.1)
                            match check_ori:
                                case 45 | 225:
                                    for i in range(1, checking_size + 1):
                                        if (
                                            checkcrossing(
                                                (checkX - i, checkY + i), netName
                                            )[1]
                                            != nType
                                            or checkcrossing(
                                                (checkX + i, checkY - i), netName
                                            )[1]
                                            != nType
                                        ):
                                            return DrcResult(
                                                True,
                                                obstacle_type=nType,
                                                conflicts=violated_net,
                                            )
                                case 135 | 315:
                                    for i in range(1, checking_size + 1):
                                        if (
                                            checkcrossing(
                                                (checkX + i, checkY + i), netName
                                            )[1]
                                            != nType
                                            or checkcrossing(
                                                (checkX - i, checkY - i), netName
                                            )[1]
                                            != nType
                                        ):
                                            return DrcResult(
                                                True,
                                                obstacle_type=nType,
                                                conflicts=violated_net,
                                            )
                            crossing_neighbor = (checkX, checkY, check_ori)
                            for _ in range(max(0, checking_size)):
                                checkX += step[0]
                                checkY += step[1]
                                crossing_neighbor = (checkX, checkY, check_ori)
                                cr_bviolated, cr_nType, cr_nNet = checkcrossing(
                                    crossing_neighbor, netName
                                )
                                if cr_bviolated:
                                    if cr_nType == "port" and cr_nNet == netName:
                                        continue
                                    else:
                                        return DrcResult(
                                            True,
                                            obstacle_type=nType,
                                            conflicts=violated_net,
                                        )
                            return DrcResult(
                                True, crossing_neighbor, "crossing_45", nNet
                            )
                        else:
                            return DrcResult(
                                True, obstacle_type=nType, conflicts=violated_net
                            )
                    return DrcResult(True, obstacle_type=nType, conflicts=violated_net)
            case "bend_45_1" | "bend_45_2":
                start_orientation = currentNode_pos[2]
                first_length, second_length = (
                    (self.ben_45_part1, self.ben_45_part2)
                    if nb_type == "bend_45_1"
                    else (self.ben_45_part2, self.ben_45_part1)
                )
                end, conflicts, blocked = self._scan_route_segments(
                    (checkX, checkY),
                    (
                        (
                            self.checkSteps[start_orientation],
                            first_length,
                            start_orientation,
                        ),
                        (checkstep, second_length, check_ori),
                    ),
                    netName,
                )
                if blocked:
                    return blocked
                if conflicts:
                    return DrcResult(True, conflicts=conflicts)
                if (
                    enable_prediction
                    and crossing_enable
                    and self._prediction_is_blocked(end, checkstep, check_ori, netName)
                ):
                    return DrcResult(True)
            case "bend_90":
                start_orientation = currentNode_pos[2]
                start_step = self.checkSteps[start_orientation]
                diagonal_step = (
                    start_step[0] + checkstep[0],
                    start_step[1] + checkstep[1],
                )
                end, conflicts, blocked = self._scan_route_segments(
                    (checkX, checkY),
                    (
                        (start_step, self.ben_45_part1, start_orientation),
                        (
                            diagonal_step,
                            self.radius - self.ben_45_part1,
                            (start_orientation + check_ori) / 2,
                        ),
                        (checkstep, self.ben_45_part1, check_ori),
                    ),
                    netName,
                )
                if blocked:
                    return blocked
                if conflicts:
                    return DrcResult(True, conflicts=conflicts)
                if (
                    enable_prediction
                    and crossing_enable
                    and self._prediction_is_blocked(end, checkstep, check_ori, netName)
                ):
                    return DrcResult(True)
            case "offset_1":
                straight = int(self.ben_45_part1 / 2)
                for _ in range(straight):
                    checkX += checkstep[0]
                    checkY += checkstep[1]
                    bviolated, nType, nNet = checkBitmap(
                        (checkX, checkY, check_ori), netName
                    )
                    if bviolated:
                        if nType == "port" and nNet == netName:
                            pass
                        else:
                            return DrcResult(True, obstacle_type=nType, conflicts=nNet)
                if check_ori == 0 or check_ori == 180:
                    checkY += step[1]
                else:
                    checkX += step[0]
                for _ in range(self.ben_45_part1 - straight + 1):
                    checkX += checkstep[0]
                    checkY += checkstep[1]
                    bviolated, nType, nNet = checkBitmap(
                        (checkX, checkY, check_ori), netName
                    )
                    if bviolated:
                        if nType == "port" and nNet == netName:
                            pass
                        else:
                            return DrcResult(True, obstacle_type=nType, conflicts=nNet)
            case "offset_2":
                straight = int(self.ben_45_part1 / 2)
                for _ in range(straight):
                    checkX += checkstep[0]
                    checkY += checkstep[1]
                    bviolated, nType, nNet = checkBitmap(
                        (checkX, checkY, check_ori), netName
                    )
                    if bviolated:
                        if nType == "port" and nNet == netName:
                            pass
                        else:
                            return DrcResult(True, obstacle_type=nType, conflicts=nNet)
                if check_ori == 0 or check_ori == 180:
                    checkY += step[1]
                else:
                    checkX += step[0]
                for _ in range(self.ben_45_part1 - straight + 2):
                    checkX += checkstep[0]
                    checkY += checkstep[1]
                    bviolated, nType, nNet = checkBitmap(
                        (checkX, checkY, check_ori), netName
                    )
                    if bviolated:
                        if nType == "port" and nNet == netName:
                            pass
                        else:
                            return DrcResult(True, obstacle_type=nType, conflicts=nNet)
        return DrcResult(False)

    def checkSpacing(
        self,
        neighbor,
        check_region: int,
        groups: Collection[str] = (),
    ) -> int:
        x, y, ori = neighbor.pos
        count = 0
        perpendicular = _PERPENDICULAR_STEPS.get(ori)
        if perpendicular is None:
            raise ValueError(f"Unsupported routing orientation {ori!r}")

        dx, dy = perpendicular
        for distance in range(1, check_region + 1):
            for sign in (-1, 1):
                check_x = x + sign * distance * dx
                check_y = y + sign * distance * dy
                if not self._bitmap.in_bounds(check_x, check_y):
                    count += 1
                    continue
                node = self._bitmap.node(check_x, check_y)
                if node.type in {"blk", "port"}:
                    count += 1
                elif node.netID and node.netID[0] not in groups:
                    count += 1
        return count

    def checkBitmap(self, index, check_net=None):
        """check the neighbors of the current index
        index[0]: x
        index[1]: y
        """
        _bitmap = self._bitmap
        width, height = _bitmap.width, _bitmap.height
        x, y = index[0], index[1]

        xrangmin, xrangmax, yrangmin, yrangmax = clip(x, y, height, width)

        bitMap = _bitmap.bitMap
        for i in range(xrangmin, xrangmax):
            for j in range(yrangmin, yrangmax):
                x = bitMap[i, j]
                if x.type != "empty":
                    return [True, x]
        return [False]

    def checkBitmap_ori(self, index, check_net=None):
        x, y, ori = index
        perpendicular = _PERPENDICULAR_STEPS.get(ori)
        if perpendicular is None:
            raise ValueError(f"Unsupported routing orientation {ori!r}")

        types = set()
        netID = None
        dx, dy = perpendicular
        for offset in (-1, 0, 1):
            check_x, check_y = x + offset * dx, y + offset * dy
            if not self._bitmap.in_bounds(check_x, check_y):
                return (True, "blk", None)
            node = self._bitmap.node(check_x, check_y)
            node_type = node.type
            if node_type in {"port", "blk"}:
                return (True, node_type, node.blkID)
            if node_type != "empty":
                types.add(node_type)
                netID = node.netID[0]

        if len(types) > 1:
            return (True, "compound", netID)
        if types:
            return (True, next(iter(types)), netID)
        return (False, "empty", None)

    def checkSingleNode(self, index, check_net=None):  # Acspoints
        ans = self._bitmap.bitMap[index[0], index[1]]
        match ans.type:
            case "empty":
                return (False, ans.type, None)
            case "blk":
                return (True, ans.type, ans.blkID)
            case "port":
                return (True, ans.type, ans.blkID)
            case _:  # waveguide or compound
                return (True, ans.type, ans.netID[0])

    def crossing_check(self, host_net, slave_net, straight_count, manhattan=True):
        if manhattan:
            straight_length = straight_count * self.resolution
        else:
            straight_length = straight_count * self.resolution * np.sqrt(2)
        return host_net.crossing_check(slave_net, straight_length)


def clip(x, y, height, width):
    xrangmin = min(max(x - 1, 0), width - 1)
    xrangmax = max(min(x + 2, width), 0)  # [ ... )
    yrangmin = min(max(y - 1, 0), height - 1)
    yrangmax = max(min(y + 2, height), 0)
    return xrangmin, xrangmax, yrangmin, yrangmax
