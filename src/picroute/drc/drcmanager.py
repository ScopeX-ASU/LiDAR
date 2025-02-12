import math

import numpy as np
from numba import jit
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    box,
)
from shapely.ops import unary_union

from ..database.schematic import CustomSchematic
from .bitmap import Bitmap


class DrcManager(object):
    def __init__(self, Cirdb: CustomSchematic, config):
        self._cirdb = Cirdb

        self.dieSize = Cirdb.schematic.settings.die_area  # Shape: [[],[]]
        self.resolution = config.dr.grid_resolution
        self._bitmap = Bitmap(self.dieSize, self.resolution)
        # self._rtree = Spatial()
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
                    match orient:
                        case 0:
                            port_len = len(inst.port_0)
                        case 90:
                            port_len = len(inst.port_90)
                        case 180:
                            port_len = len(inst.port_180)
                        case 270:
                            port_len = len(inst.port_270)

                    if net.NetPort1 == Port and port_len <= 2:
                        net.reverse = True
                    nets[Port.netName] = None
                    net.groups.append(name)
                    inst_group["nets"] = nets
                    inst_group["configs"] = None
                    self._cirdb._group_Nets[name] = inst_group

                    step = orientations[Port.orientation]
                    loc = (np.array(ports[0].center) / self.resolution).astype(np.int32)
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
                    mean_loc = (
                        (np.array(order_ports[0][1]) + np.array(order_ports[-1][1]))
                        / (2 * self.resolution)
                    ).astype(np.int32)
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
                                    loc = (np.array(port[1]) / self.resolution).astype(
                                        np.int32
                                    )
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

    def bViolateDRC(
        self, currentNode, step, crossing_enable, netName, enable_prediction
    ):  # crossing enable
        currentNode_pos = currentNode.pos
        crossing_budget = currentNode._crossing_budget
        straight_count = currentNode.straightCount
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
                        return (False, None)
                    elif nType in {"blk", "compound", "port"}:
                        return (True, None, nType, violated_net)
                    elif abs(check_ori - nType) == 90:
                        if (
                            not crossing_enable
                            or crossing_budget <= 0
                            or dbNets[nNet].current_budget <= 0
                        ):
                            return (True, None, nType, violated_net)
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
                                        return (True, None, nType, violated_net)
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
                                        return (True, None, nType, violated_net)

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
                                        return (True, None, nType, violated_net)

                            return (True, crossing_neighbor, "crossing_0", nNet)
                        else:
                            return (True, None, nType, violated_net)
                    return (True, None, nType, violated_net)

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
                        return (False, None)
                    if nType in {"blk", "compound", "port"}:
                        return (True, None, nType, violated_net)
                    if (
                        check_ori + nType
                    ) % 180 == 0:  # 45+135 / 225+135 / 135+45 / 315+45
                        if (
                            not crossing_enable
                            or crossing_budget <= 0
                            or dbNets[nNet].current_budget <= 0
                        ):
                            return (True, None, nType, violated_net)
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
                                            return (True, None, nType, violated_net)
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
                                            return (True, None, nType, violated_net)
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
                                        return (True, None, nType, violated_net)
                            return (True, crossing_neighbor, "crossing_45", nNet)
                        else:
                            return (True, None, nType, violated_net)
                    return (True, None, nType, violated_net)
            case "bend_45_2":
                violated_net = set()
                ori1 = currentNode_pos[2]
                ori2 = check_ori
                checkstep1 = self.checkSteps[ori1]
                checkstep2 = checkstep
                # check_len = self.straight_45_part2
                for _ in range(self.ben_45_part2):
                    checkX += checkstep1[0]
                    checkY += checkstep1[1]
                    bviolated, nType, nNet = checkBitmap(
                        (checkX, checkY, ori1), netName
                    )
                    if bviolated:
                        if nType == "port" and nNet == netName:
                            continue
                        elif nType in {"port", "blk"}:
                            return (True, None, nType, nNet)
                        else:
                            violated_net.add(nNet)
                for _ in range(self.ben_45_part1):
                    checkX += checkstep2[0]
                    checkY += checkstep2[1]
                    bviolated, nType, nNet = checkBitmap(
                        (checkX, checkY, ori2), netName
                    )
                    if bviolated:
                        if nType == "port" and nNet == netName:
                            continue
                        elif nType in {"port", "blk"}:
                            return (True, None, nType, nNet)
                        else:
                            violated_net.add(nNet)

                # Prediction

                if not violated_net and enable_prediction and crossing_enable:
                    try:
                        tempX, tempY = checkX, checkY
                        for _ in range(self.predict_length):
                            tempX += checkstep2[0]
                            tempY += checkstep2[1]
                            bviolated, nType, nNet = checkBitmap(
                                (tempX, tempY, check_ori), netName
                            )
                            if bviolated:
                                if nType == "port" and nNet == netName:
                                    continue
                                else:
                                    return (True, None, None, None)
                    except:
                        return (True, None, None, None)
                elif violated_net:
                    return (True, None, None, violated_net)
            case "bend_45_1":
                violated_net = set()
                ori1 = currentNode_pos[2]
                ori2 = check_ori
                checkstep1 = self.checkSteps[ori1]
                checkstep2 = checkstep
                for _ in range(self.ben_45_part1):
                    checkX += checkstep1[0]
                    checkY += checkstep1[1]
                    bviolated, nType, nNet = checkBitmap(
                        (checkX, checkY, ori1), netName
                    )
                    if bviolated:
                        if nType == "port" and nNet == netName:
                            continue
                        elif nType in {"port", "blk"}:
                            return (True, None, nType, nNet)
                        else:
                            violated_net.add(nNet)
                for _ in range(self.ben_45_part2):
                    checkX += checkstep2[0]
                    checkY += checkstep2[1]
                    bviolated, nType, nNet = checkBitmap(
                        (checkX, checkY, ori2), netName
                    )
                    if bviolated:
                        if nType == "port" and nNet == netName:
                            continue
                        elif nType in {"port", "blk"}:
                            return (True, None, nType, nNet)
                        else:
                            violated_net.add(nNet)

                # Prediction

                if not violated_net and enable_prediction and crossing_enable:
                    try:
                        tempX, tempY = checkX, checkY
                        for _ in range(self.predict_length):
                            tempX += checkstep2[0]
                            tempY += checkstep2[1]
                            bviolated, nType, nNet = checkBitmap(
                                (tempX, tempY, check_ori), netName
                            )
                            if bviolated:
                                if nType == "port" and nNet == netName:
                                    continue
                                else:
                                    return (True, None, None, None)
                    except:
                        return (True, None, None, None)
                elif violated_net:
                    return (True, None, None, violated_net)
            case "bend_90":
                violated_net = set()
                ori1 = currentNode_pos[2]
                ori2 = check_ori
                checkstep1 = self.checkSteps[ori1]
                checkstep2 = checkstep
                for _ in range(self.ben_45_part1):
                    checkX += checkstep1[0]
                    checkY += checkstep1[1]
                    bviolated, nType, nNet = checkBitmap(
                        (checkX, checkY, ori1), netName
                    )
                    if bviolated:
                        if nType == "port" and nNet == netName:
                            continue
                        elif nType in {"port", "blk"}:
                            return (True, None, nType, nNet)
                        else:
                            violated_net.add(nNet)
                ori3 = (ori1 + ori2) / 2
                for _ in range(self.radius - self.ben_45_part1):
                    checkX += checkstep1[0] + checkstep2[0]
                    checkY += checkstep1[1] + checkstep2[1]
                    bviolated, nType, nNet = checkBitmap(
                        (checkX, checkY, ori3), netName
                    )
                    if bviolated:
                        if nType == "port" and nNet == netName:
                            continue
                        elif nType in {"port", "blk"}:
                            return (True, None, nType, nNet)
                        else:
                            violated_net.add(nNet)
                for _ in range(self.ben_45_part1):
                    checkX += checkstep2[0]
                    checkY += checkstep2[1]
                    bviolated, nType, nNet = checkBitmap(
                        (checkX, checkY, ori2), netName
                    )
                    if bviolated:
                        if nType == "port" and nNet == netName:
                            continue
                        elif nType in {"port", "blk"}:
                            return (True, None, nType, nNet)
                        else:
                            violated_net.add(nNet)
                # prediction
                if not violated_net and enable_prediction and crossing_enable:
                    try:
                        for _ in range(self.predict_length):
                            checkX += checkstep2[0]
                            checkY += checkstep2[1]
                            bviolated, nType, nNet = checkBitmap(
                                (checkX, checkY, ori2), netName
                            )
                            if bviolated:
                                if nType == "port" and nNet == netName:
                                    continue
                                else:
                                    return (True, None, None, None)
                    except:
                        return (True, None, None, None)
                elif violated_net:
                    return (True, None, None, violated_net)
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
                            return (True, None, nType, nNet)
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
                            return (True, None, nType, nNet)
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
                            return (True, None, nType, nNet)
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
                            return (True, None, nType, nNet)
        return (False, None)

    def bViolateDRC_orig(
        self, currentNode, step, crossing_enable, netName, enable_prediction
    ):  # crossing enable
        currentNode_pos = currentNode.pos
        crossing_budget = currentNode._crossing_budget
        straight_count = currentNode.straightCount
        check_ori = step[2]  # next step orientation
        checkstep = self.checkSteps[check_ori]
        checkX = currentNode_pos[0]
        checkY = currentNode_pos[1]
        nb_type = step[3]  # check type

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
                    violated_net = set()
                    violated_net.add(nNet)
                    if nType == "port" and nNet == netName:
                        return (False, None)
                    elif nType in {"blk", "compound", "port"}:
                        return (True, None, nType, violated_net)
                    elif abs(check_ori - nType) == 90:
                        if (
                            not crossing_enable
                            or crossing_budget <= 0
                            or self._cirdb.dbNets[nNet].current_budget <= 0
                        ):
                            return (True, None, nType, violated_net)
                        # 90+180 / 270+180 / 0+90 / 180+90
                        host = self._cirdb.dbNets[netName]
                        slave = self._cirdb.dbNets[nNet]
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
                                        return (True, None, nType, violated_net)
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
                                        return (True, None, nType, violated_net)

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
                                        return (True, None, nType, violated_net)

                            return (True, crossing_neighbor, "crossing_0", nNet)
                        else:
                            return (True, None, nType, violated_net)
                    return (True, None, nType, violated_net)

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
                        return (False, None)
                    if nType in {"blk", "compound", "port"}:
                        return (True, None, nType, violated_net)
                    if (
                        check_ori + nType
                    ) % 180 == 0:  # 45+135 / 225+135 / 135+45 / 315+45
                        if (
                            not crossing_enable
                            or crossing_budget <= 0
                            or self._cirdb.dbNets[nNet].current_budget <= 0
                        ):
                            return (True, None, nType, violated_net)
                        host = self._cirdb.dbNets[netName]
                        slave = self._cirdb.dbNets[nNet]
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
                                            return (True, None, nType, violated_net)
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
                                            return (True, None, nType, violated_net)
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
                                        return (True, None, nType, violated_net)
                            return (True, crossing_neighbor, "crossing_45", nNet)
                        else:
                            return (True, None, nType, violated_net)
                    return (True, None, nType, violated_net)
            case "bend_45_2":
                violated_net = set()
                ori1 = currentNode_pos[2]
                ori2 = check_ori
                checkstep1 = self.checkSteps[ori1]
                checkstep2 = checkstep
                for _ in range(self.ben_45_part2):
                    checkX += checkstep1[0]
                    checkY += checkstep1[1]
                    bviolated, nType, nNet = checkBitmap(
                        (checkX, checkY, ori1), netName
                    )
                    if bviolated:
                        if nType == "port" and nNet == netName:
                            continue
                        elif nType in {"port", "blk"}:
                            return (True, None, nType, nNet)
                        else:
                            violated_net.add(nNet)
                for _ in range(self.ben_45_part1):
                    checkX += checkstep2[0]
                    checkY += checkstep2[1]
                    bviolated, nType, nNet = checkBitmap(
                        (checkX, checkY, ori2), netName
                    )
                    if bviolated:
                        if nType == "port" and nNet == netName:
                            continue
                        elif nType in {"port", "blk"}:
                            return (True, None, nType, nNet)
                        else:
                            violated_net.add(nNet)

                # Prediction

                if not violated_net and enable_prediction and crossing_enable:
                    try:
                        tempX, tempY = checkX, checkY
                        for _ in range(self.predict_length):
                            tempX += checkstep2[0]
                            tempY += checkstep2[1]
                            bviolated, nType, nNet = checkBitmap(
                                (tempX, tempY, check_ori), netName
                            )
                            if bviolated:
                                if nType == "port" and nNet == netName:
                                    continue
                                else:
                                    return (True, None, None, None)
                    except:
                        return (True, None, None, None)
                elif violated_net:
                    return (True, None, None, violated_net)
            case "bend_45_1":
                violated_net = set()
                ori1 = currentNode_pos[2]
                ori2 = check_ori
                checkstep1 = self.checkSteps[ori1]
                checkstep2 = checkstep
                for _ in range(self.ben_45_part1):
                    checkX += checkstep1[0]
                    checkY += checkstep1[1]
                    bviolated, nType, nNet = checkBitmap(
                        (checkX, checkY, ori1), netName
                    )
                    if bviolated:
                        if nType == "port" and nNet == netName:
                            continue
                        elif nType in {"port", "blk"}:
                            return (True, None, nType, nNet)
                        else:
                            violated_net.add(nNet)
                for _ in range(self.ben_45_part2):
                    checkX += checkstep2[0]
                    checkY += checkstep2[1]
                    bviolated, nType, nNet = checkBitmap(
                        (checkX, checkY, ori2), netName
                    )
                    if bviolated:
                        if nType == "port" and nNet == netName:
                            continue
                        elif nType in {"port", "blk"}:
                            return (True, None, nType, nNet)
                        else:
                            violated_net.add(nNet)

                # Prediction

                if not violated_net and enable_prediction and crossing_enable:
                    try:
                        tempX, tempY = checkX, checkY
                        for _ in range(self.predict_length):
                            tempX += checkstep2[0]
                            tempY += checkstep2[1]
                            bviolated, nType, nNet = checkBitmap(
                                (tempX, tempY, check_ori), netName
                            )
                            if bviolated:
                                if nType == "port" and nNet == netName:
                                    continue
                                else:
                                    return (True, None, None, None)
                    except:
                        return (True, None, None, None)
                elif violated_net:
                    return (True, None, None, violated_net)
            case "bend_90":
                violated_net = set()
                ori1 = currentNode_pos[2]
                ori2 = check_ori
                checkstep1 = self.checkSteps[ori1]
                checkstep2 = checkstep
                for _ in range(self.ben_45_part1):
                    checkX += checkstep1[0]
                    checkY += checkstep1[1]
                    bviolated, nType, nNet = checkBitmap(
                        (checkX, checkY, ori1), netName
                    )
                    if bviolated:
                        if nType == "port" and nNet == netName:
                            continue
                        elif nType in {"port", "blk"}:
                            return (True, None, nType, nNet)
                        else:
                            violated_net.add(nNet)
                ori3 = (ori1 + ori2) / 2
                for _ in range(self.radius - self.ben_45_part1):
                    checkX += checkstep1[0] + checkstep2[0]
                    checkY += checkstep1[1] + checkstep2[1]
                    bviolated, nType, nNet = checkBitmap(
                        (checkX, checkY, ori3), netName
                    )
                    if bviolated:
                        if nType == "port" and nNet == netName:
                            continue
                        elif nType in {"port", "blk"}:
                            return (True, None, nType, nNet)
                        else:
                            violated_net.add(nNet)
                for _ in range(self.ben_45_part1):
                    checkX += checkstep2[0]
                    checkY += checkstep2[1]
                    bviolated, nType, nNet = checkBitmap(
                        (checkX, checkY, ori2), netName
                    )
                    if bviolated:
                        if nType == "port" and nNet == netName:
                            continue
                        elif nType in {"port", "blk"}:
                            return (True, None, nType, nNet)
                        else:
                            violated_net.add(nNet)
                # prediction
                if not violated_net and enable_prediction and crossing_enable:
                    try:
                        for _ in range(self.predict_length):
                            checkX += checkstep2[0]
                            checkY += checkstep2[1]
                            bviolated, nType, nNet = checkBitmap(
                                (checkX, checkY, ori2), netName
                            )
                            if bviolated:
                                if nType == "port" and nNet == netName:
                                    continue
                                else:
                                    return (True, None, None, None)
                    except:
                        return (True, None, None, None)
                elif violated_net:
                    return (True, None, None, violated_net)
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
                            return (True, None, nType, nNet)
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
                            return (True, None, nType, nNet)
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
                            return (True, None, nType, nNet)
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
                            return (True, None, nType, nNet)
        return (False, None)

    def checkSpacing(self, neighbor, check_region, groups={}):
        x, y, ori = neighbor.pos
        self._bitmap
        bitMap = self._bitmap.bitMap
        count = 0
        height, width = self.height, self.width

        match ori:
            case 0 | 180:
                for i in range(1, check_region + 1):
                    checkx, checky = clip2(x, y + i, height, width)
                    ans = bitMap[checkx, checky]
                    nets = ans.netID
                    if ans.type in {"blk", "port"}:
                        count += 1
                    elif nets:
                        if nets[0] in groups:
                            pass
                        else:
                            count += 1
                    checkx, checky = clip2(x, y - i, height, width)
                    ans = bitMap[checkx, checky]
                    nets = ans.netID
                    if ans.type in {"blk", "port"}:
                        count += 1
                    elif nets:
                        if nets[0] in groups:
                            pass
                        else:
                            count += 1
                return count

            case 90 | 270:
                for i in range(1, check_region + 1):
                    checkx, checky = clip2(x + i, y, height, width)
                    ans = bitMap[checkx, checky]
                    type = ans.type
                    if type != "empty":
                        if type in {"port", "blk"}:
                            count += 1
                        else:
                            netID = ans.netID[0]
                            if netID in groups:
                                pass
                            else:
                                count += 1
                    checkx, checky = clip2(x - i, y, height, width)
                    ans = bitMap[checkx, checky]
                    type = ans.type
                    if type != "empty":
                        if type in {"port", "blk"}:
                            count += 1
                        else:
                            netID = ans.netID[0]
                            if netID in groups:
                                pass
                            else:
                                count += 1
                return count

            case 45 | 225:
                for i in range(1, check_region + 1):
                    checkx, checky = clip2(x + i, y - i, height, width)
                    ans = bitMap[checkx, checky]
                    nets = ans.netID
                    if ans.type in {"blk", "port"}:
                        count += 1
                    elif nets:
                        if nets[0] in groups:
                            pass
                        else:
                            count += 1
                    checkx, checky = clip2(x - i, y + i, height, width)
                    ans = bitMap[checkx, checky]
                    nets = ans.netID
                    if ans.type in {"blk", "port"}:
                        count += 1
                    elif nets:
                        if nets[0] in groups:
                            pass
                        else:
                            count += 1
                return count

            case 135 | 315:
                for i in range(1, check_region + 1):
                    checkx, checky = clip2(x + i, y + i, height, width)
                    ans = bitMap[checkx, checky]
                    nets = ans.netID
                    if ans.type in {"blk", "port"}:
                        count += 1
                    elif nets:
                        if nets[0] in groups:
                            pass
                        else:
                            count += 1
                    checkx, checky = clip2(x - i, y - i, height, width)
                    ans = bitMap[checkx, checky]
                    nets = ans.netID
                    if ans.type in {"blk", "port"}:
                        count += 1
                    elif nets:
                        if nets[0] in groups:
                            pass
                        else:
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
        _bitmap = self._bitmap
        bitMap = _bitmap.bitMap
        ori = index[2]
        ans = None
        types = set()
        netID = None
        match ori:
            case 0:
                x, y = index[0], index[1] - 1
                for i in range(0, 3):
                    ans = bitMap[x, y + i]
                    type = ans.type
                    if type != "empty":
                        if type in {"port", "blk"}:
                            return (True, ans.type, ans.blkID)
                        else:
                            types.add(ans.type)
                            netID = ans.netID[0]

                types_len = len(types)
                if types_len > 1:
                    return (True, "compound", netID)
                elif types_len == 1:
                    for type in types:
                        return (True, type, netID)
            case 45:
                x, y = index[0] - 1, index[1] + 1
                for i in range(0, 3):
                    ans = bitMap[x + i, y - i]
                    type = ans.type
                    if type != "empty":
                        if type in {"port", "blk"}:
                            return (True, ans.type, ans.blkID)
                        else:
                            types.add(ans.type)
                            netID = ans.netID[0]
                types_len = len(types)
                if types_len > 1:
                    return (True, "compound", netID)
                elif types_len == 1:
                    for type in types:
                        return (True, type, netID)
            case 90:
                x, y = index[0] - 1, index[1]
                for i in range(0, 3):
                    ans = bitMap[x + i, y]
                    type = ans.type
                    if type != "empty":
                        if type in {"port", "blk"}:
                            return (True, ans.type, ans.blkID)
                        else:
                            types.add(ans.type)
                            netID = ans.netID[0]
                types_len = len(types)
                if types_len > 1:
                    return (True, "compound", netID)
                elif types_len == 1:
                    for type in types:
                        return (True, type, netID)
            case 135:
                x, y = index[0] - 1, index[1] - 1
                for i in range(0, 3):
                    ans = bitMap[x + i, y + i]
                    type = ans.type
                    if type != "empty":
                        if type in {"port", "blk"}:
                            return (True, ans.type, ans.blkID)
                        else:
                            types.add(ans.type)
                            netID = ans.netID[0]
                types_len = len(types)
                if types_len > 1:
                    return (True, "compound", netID)
                elif types_len == 1:
                    for type in types:
                        return (True, type, netID)
            case 180:
                x, y = index[0], index[1] - 1
                for i in range(0, 3):
                    ans = bitMap[x, y + i]
                    type = ans.type
                    if type != "empty":
                        if type in {"port", "blk"}:
                            return (True, ans.type, ans.blkID)
                        else:
                            types.add(ans.type)
                            netID = ans.netID[0]
                types_len = len(types)
                if types_len > 1:
                    return (True, "compound", netID)
                elif types_len == 1:
                    for type in types:
                        return (True, type, netID)
            case 225:
                x, y = index[0] - 1, index[1] + 1
                for i in range(0, 3):
                    ans = bitMap[x + i, y - i]
                    type = ans.type
                    if type != "empty":
                        if type in {"port", "blk"}:
                            return (True, ans.type, ans.blkID)
                        else:
                            types.add(ans.type)
                            netID = ans.netID[0]
                types_len = len(types)
                if types_len > 1:
                    return (True, "compound", netID)
                elif types_len == 1:
                    for type in types:
                        return (True, type, netID)
            case 270:
                x, y = index[0] - 1, index[1]
                for i in range(0, 3):
                    ans = bitMap[x + i, y]
                    type = ans.type
                    if type != "empty":
                        if type in {"port", "blk"}:
                            return (True, ans.type, ans.blkID)
                        else:
                            types.add(ans.type)
                            netID = ans.netID[0]
                types_len = len(types)
                if types_len > 1:
                    return (True, "compound", netID)
                elif types_len == 1:
                    for type in types:
                        return (True, type, netID)
            case 315:
                x, y = index[0] - 1, index[1] - 1
                for i in range(0, 3):
                    ans = bitMap[x + i, y + i]
                    type = ans.type
                    if type != "empty":
                        if type in {"port", "blk"}:
                            return [True, ans.type, ans.blkID]
                        else:
                            types.add(ans.type)
                            netID = ans.netID[0]
                types_len = len(types)
                if types_len > 1:
                    return (True, "compound", netID)
                elif types_len == 1:
                    for type in types:
                        return (True, type, netID)
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

    # Final component DRC
    def componentDRC(self, rectangles, instanceName: str, netName):
        for rects in rectangles:
            for rect in rects:
                x, y = int(rect[0]), int(rect[1])
                if (
                    self._bitmap.bitMap[x][y].type == "blk"
                    and self._bitmap.bitMap[x][y].blkID == instanceName
                ):
                    continue
                elif (
                    self._bitmap.bitMap[x][y].type == "port"
                    and self._bitmap.bitMap[x][y].blkID == netName
                ):
                    continue
                if self._bitmap.bitMap[x][y].type != "empty":
                    return True
        return False


@jit(nopython=True)
def clip(x, y, height, width):
    xrangmin = min(max(x - 1, 0), width - 1)
    xrangmax = max(min(x + 2, width), 0)  # [ ... )
    yrangmin = min(max(y - 1, 0), height - 1)
    yrangmax = max(min(y + 2, height), 0)
    return xrangmin, xrangmax, yrangmin, yrangmax


@jit(nopython=True)
def clip2(x, y, height, width):
    checkx = min(max(x, 0), width - 1)
    checky = min(max(y, 0), height - 1)
    return checkx, checky


def is_curve_section(intersection):
    """
    check the rectangle type
    """
    if isinstance(intersection, GeometryCollection):  # complicated case
        return "bend"
    elif isinstance(intersection, MultiLineString):
        for line in intersection.geoms:
            start = line.coords[0]
            end = line.coords[1]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            if dx == 0:
                return "straight90"
            if dy == 0:
                return "straight0"
            slope = dy / dx
            if abs(slope - 1) < 1e-3:
                return "straight45"
            elif abs(slope + 1) < 1e-3:
                return "straight135"
            elif abs(slope) == float("inf") or abs(slope) > 5000:
                return "straight90"
            elif abs(slope) < 1e-3:
                return "straight0"
            return "bend"
    elif isinstance(intersection, LineString):
        start = intersection.coords[0]
        end = intersection.coords[1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        if dx == 0:
            return "straight90"
        if dy == 0:
            return "straight0"
        slope = dy / dx
        if abs(slope - 1) < 1e-3:
            return "straight45"
        elif abs(slope + 1) < 1e-3:
            return "straight135"
        elif slope == float("inf") or abs(slope) > 5000:
            return "straight90"
        elif abs(slope) < 1e-3:
            return "straight0"
        return "bend"
    else:
        assert 0
        # return "bend"


def split_polygon_to_rectangles(polygon, grid_size=1, DRC: bool = False):
    """
    polygon is splited for DRC check if DRC == True, otherwise it will be used to update the bitmap
    rectangle type: {bend, straight0, straight45, straight90, straight135, straight180}
    """

    minx, miny, maxx, maxy = polygon.bounds
    if not DRC:
        minx, miny, maxx, maxy = int(minx), int(miny), int(maxx) + 1, int(maxy) + 1
    else:
        minx, miny, maxx, maxy = (
            int(minx) + 1,
            int(miny) + 1,
            int(maxx) - 1,
            int(maxy) - 1,
        )

    curve_rects = []
    straight_0 = []
    straight_45 = []
    straight_90 = []
    straight_135 = []

    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            rect = box(x, y, x + grid_size, y + grid_size)
            intersection = rect.intersection(polygon)
            if not intersection.is_empty:
                type = is_curve_section(intersection)
                if type == "bend":
                    curve_rects.append(rect)
                elif type == "straight90":
                    straight_90.append(rect)
                elif type == "straight0":
                    straight_0.append(rect)
                elif type == "straight135":
                    straight_135.append(rect)
                else:
                    straight_45.append(rect)
            y += grid_size
        x += grid_size

    return curve_rects, straight_90, straight_0, straight_45, straight_135


def merge_rectangles(rectangles):
    def is_aligned_and_adjacent(rect1, rect2):
        # Check if two rectangles are aligned and adjacent
        return (
            (rect1.bounds[2] == rect2.bounds[0] or rect1.bounds[0] == rect2.bounds[2])
            and rect1.bounds[1] == rect2.bounds[1]
            and rect1.bounds[3] == rect2.bounds[3]
        ) or (
            (rect1.bounds[3] == rect2.bounds[1] or rect1.bounds[1] == rect2.bounds[3])
            and rect1.bounds[0] == rect2.bounds[0]
            and rect1.bounds[2] == rect2.bounds[2]
        )

    merged_rectangles = []
    while rectangles:
        rect = rectangles.pop()
        to_merge = [rect]
        found = True

        while found:
            found = False
            for other_rect in rectangles:
                if is_aligned_and_adjacent(rect, other_rect):
                    to_merge.append(other_rect)
                    rectangles.remove(other_rect)
                    rect = unary_union(to_merge)
                    found = True
                    break

        merged_rectangles.append(rect)

    return merged_rectangles
