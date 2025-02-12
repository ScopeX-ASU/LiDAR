# from pyutils.general import logger, TimerCtx
import gdsfactory as gf
import numpy as np
from shapely.geometry import LineString, Point

from picroute.database.schematic import CustomSchematic, Nets
from picroute.drc.drcmanager import DrcManager
from picroute.utils.general import logger
from picroute.utils.route_bundle_sbend import route_bundle_sbend
from picroute.utils.smooth import alignment, smooth

from ..queue.pqueue import AdvancedPriorityQueue


class DrGridRoute(object):
    def __init__(self, CirDB: CustomSchematic, DrcMgr: DrcManager, config):
        self._cirdb = CirDB
        self._drcmgr = DrcMgr
        self._config = config

        self.global_failedNet = set()
        self.historyMap = np.zeros(
            (self._drcmgr.bitMap_width, self._drcmgr.bitMap_height, 8), dtype=np.int32
        )
        self.orientations = {0: 0, 45: 1, 90: 2, 135: 3, 180: 4, 225: 5, 270: 6, 315: 7}
        self._ripup_times = 0  # config.dr.ripup_times

        self.prop = self._config.dr.loss_propagation
        self.cross = self._config.il_cross

    def solve(self) -> bool:
        bSuccess = True

        self.process_net_order()

        self._drcmgr.initDRC()
        if self._config.dr.group:
            self._drcmgr.initPorts()
        else:
            self._drcmgr.bitMap.initPorts(self._cirdb.dbNets)

        # initialize net routing priority queue
        self.net_PQ = AdvancedPriorityQueue()
        self.unrouted = 0

        # To do:
        # add unrouted nets to pq

        self.addUnroutedNetsToPQ()

        bSuccess = self.solveDR()

        if bSuccess:
            """ For the last iteration with vionets
            """
            for _, value in self._cirdb._dbNets.items():
                if value.vionets == 0:
                    value.crossing_num = len(value.crossing_nets)
                else:
                    value.crossing_num = value.vionets
            for netID in self.global_failedNet:
                net = self._cirdb.dbNets[netID]
                net.crossing_num += 1

            self.post_processing()
            self.evaluation()
        else:
            self.checkFailed()
            self.evaluation()

        return bSuccess

    def process_net_order(self):
        if self._config.dr.net_order == "topo":
            dbNets = self._cirdb.dbNets

            CR = 0
            for level, nets in enumerate(self._cirdb.topology_orders):
                linestrings = {}
                for net_name in nets:
                    net = dbNets[net_name]
                    net.comp_dist = level
                    current_line = LineString([net.port1.center, net.port2.center])
                    for line_name, line in linestrings.items():
                        intersection = current_line.intersection(line)
                        if not intersection.is_empty:
                            dbNets[net_name].topology_crossing += 1
                            dbNets[line_name].topology_crossing += 1
                            CR += 1
                    linestrings[net_name] = current_line
            print(f"==============CR: {CR}=================")

    def addUnroutedNetsToPQ(self):
        for _, value in self._cirdb._dbNets.items():
            if not value.routed and not self.net_PQ.exist(value):
                self.net_PQ.put(value)

    def solveDR(self):
        # routing with ripup and reroute
        if self.runNRR(self.net_PQ, self._config.dr.maxIteration):
            logger.info(f"{self.__class__.__name__} succeed")
            return True
        else:
            logger.warning(f"{self.__class__.__name__} fail")
            return False

    def runNRR(self, net_pq, maxIteration):
        for iter in range(maxIteration):
            logger.info(
                f"{self.__class__.__name__}: Iteration: {iter + 1:4d} / {maxIteration:4d}, Unrouted nets: Todo"
            )
            if self._config.show_temp:
                self.show_templayout()

            while not net_pq.empty():
                cur_net = net_pq.get()
                if cur_net.routed:
                    continue
                if self._config.dr.group:
                    group_len = 0
                    for groupID in cur_net.groups:
                        nets = self._cirdb._group_Nets[groupID]["nets"]
                        group_len = max(group_len, len(nets))
                    for groupID in cur_net.groups:
                        local_PQ = AdvancedPriorityQueue()
                        group = self._cirdb._group_Nets[groupID]
                        nets = group["nets"]
                        for i, netName in enumerate(nets.keys()):
                            net = self._cirdb._dbNets[netName]
                            net.routing_order = i
                            local_PQ.put(net)
                        while not local_PQ.empty():
                            local_net = local_PQ.get()
                            self.unrouted = group_len - local_net.routing_order + 1
                            if local_net.routed:
                                continue
                            bSuccess = self.routeSingleNet(
                                local_net, strictDRC=True, net_config=None, groups=nets
                            )
                            if not bSuccess:
                                bSuccess = self.routeSingleNet(
                                    local_net,
                                    strictDRC=False,
                                    net_config=None,
                                    groups=nets,
                                )
                                if not bSuccess:
                                    self._cirdb.dbNets.pop(local_net.netName)
                                    self._cirdb.abNets[local_net.netName] = local_net
                else:
                    bSuccess = self.routeSingleNet(cur_net, strictDRC=True)
                    if not bSuccess:
                        bSuccess = self.routeSingleNet(
                            cur_net, strictDRC=False
                        )  # routing with DRC relaxation
                        if not bSuccess:  # abnormal nets
                            self._cirdb.dbNets.pop(cur_net.netName)
                            self._cirdb.abNets[cur_net.netName] = cur_net
            if iter == maxIteration - 1 and len(self.global_failedNet) != 0:
                return False
            self.ripupfailedNets()
            self.addUnroutedNetsToPQ()
            if net_pq.empty():
                return True
        return False

    def show_templayout(self):
        self._cirdb.tempLayout = self._cirdb.layout.copy()
        for netID, net in self._cirdb.dbNets.items():
            if net.routed:
                for ref in net.wg_component:
                    self._cirdb.tempLayout.add_ref(ref)
        self._cirdb.tempLayout.show()

    def routeSingleNet(
        self, cur_net: Nets, strictDRC=True, net_config=None, groups=None
    ):
        from .astarsearch import AstarSearch

        print(
            f"route net: {cur_net.netName}, crossing_budget = {cur_net.crossing_budget}, strictDRC: {strictDRC}"
        )

        if self._config.dr.net_order == "topo":
            kernel = AstarSearch(
                self,
                self._cirdb,
                cur_net,
                self._drcmgr,
                self._config,
                bStrictDRC=strictDRC,
                net_config=net_config,
                groups=groups,
            )
            bSuccess = kernel.route()
            if bSuccess:
                self.registar_net(cur_net)
        else:
            if self._config.dr.group:
                if strictDRC:
                    kernel1 = AstarSearch(
                        self,
                        self._cirdb,
                        cur_net,
                        self._drcmgr,
                        self._config,
                        bStrictDRC=strictDRC,
                        net_config=net_config,
                        groups=groups,
                    )
                    bSuccess1 = kernel1.route()
                    if bSuccess1:
                        if (
                            cur_net.crossing_num == 0
                        ):  ## do not use crossing, may due to detour
                            self.registar_net(cur_net)
                            return True
                        # check crossing net: 1. distance and 2. failed_count
                        clear = False
                        for netID in cur_net.crossing_nets:
                            net = self._cirdb.dbNets[netID]
                            if cur_net.failed_count == 0 and net.distance["Euler"] > (
                                cur_net.distance["Euler"] + 1000
                            ):
                                clear = True
                                break
                        if clear:
                            bkp = cur_net.backup_net()
                            cur_net.clearNet(self._cirdb)
                            self.ripuplocalnets(bkp[1])
                            # cur_net.failed_count += 1
                            cur_net.crossing_budget = 10
                            cur_net.current_budget = 10
                            kernel1 = AstarSearch(
                                self,
                                self._cirdb,
                                cur_net,
                                self._drcmgr,
                                self._config,
                                bStrictDRC=strictDRC,
                                net_config=net_config,
                                groups=groups,
                            )
                            bSuccess1 = kernel1.route()
                            self.registar_net(cur_net)
                            return True
                        else:
                            # do not ripup
                            loss1 = self.eval_net(cur_net)
                            bkp = cur_net.backup_net()
                            cur_net.clearNet(self._cirdb, increase=False)
                    else:
                        return False

                    # have used crossing, route with crossing budget == 0
                    cur_net.crossing_budget = 0
                    cur_net.current_budget = 0
                    kernel2 = AstarSearch(
                        self,
                        self._cirdb,
                        cur_net,
                        self._drcmgr,
                        self._config,
                        bStrictDRC=strictDRC,
                        net_config=net_config,
                        groups=groups,
                    )
                    bSuccess2 = kernel2.route()
                    if bSuccess2:
                        # route successfully
                        loss2 = self.eval_net(cur_net)
                        if loss1 + 0.2 > loss2:
                            # do not detour, use no crossing solution
                            cur_net.crossing_budget = 10
                            cur_net.current_budget = 10
                            self.registar_net(cur_net)
                        else:
                            # too much detour, use the crossing solution
                            cur_net.clearNet(self._cirdb, increase=False)
                            cur_net.crossing_budget = 10
                            cur_net.current_budget = 10
                            kernel1 = AstarSearch(
                                self,
                                self._cirdb,
                                cur_net,
                                self._drcmgr,
                                self._config,
                                bStrictDRC=strictDRC,
                                net_config=net_config,
                                groups=groups,
                            )
                            bSuccess1 = kernel1.route()
                            self.registar_net(cur_net)
                    else:
                        # fail to route with crossing disable
                        if bkp[0] == 0:
                            # first time, ripup the crossing nets
                            self.ripuplocalnets(bkp[1])
                            cur_net.failed_count += 1
                        cur_net.clearNet(self._cirdb, increase=False)
                        cur_net.crossing_budget = 10
                        cur_net.current_budget = 10
                        kernel1 = AstarSearch(
                            self,
                            self._cirdb,
                            cur_net,
                            self._drcmgr,
                            self._config,
                            bStrictDRC=strictDRC,
                            net_config=net_config,
                            groups=groups,
                        )
                        bSuccess1 = kernel1.route()
                        cur_net.crossing_budget = 10
                        cur_net.current_budget = 10
                        self.registar_net(cur_net)

                    return True
                else:
                    kernel = AstarSearch(
                        self,
                        self._cirdb,
                        cur_net,
                        self._drcmgr,
                        self._config,
                        bStrictDRC=strictDRC,
                        net_config=net_config,
                        groups=groups,
                    )
                    bSuccess = kernel.route()
                    if bSuccess:
                        self.registar_net(cur_net)
            else:
                kernel = AstarSearch(
                    self,
                    self._cirdb,
                    cur_net,
                    self._drcmgr,
                    self._config,
                    bStrictDRC=strictDRC,
                    net_config=net_config,
                    groups=groups,
                )
                bSuccess = kernel.route()
                if bSuccess:
                    self.registar_net(cur_net)

        return bSuccess

    def ripuplocalnets(self, cur_net_crossing):
        bitmap = self._drcmgr.bitMap.bitMap
        for netID in cur_net_crossing:
            net = self._cirdb.dbNets[netID]
            for x, y in net.rwguide:
                bitmap[x][y].delete(netID)
            net.clearNet(self._cirdb)
        if self._config.show_temp:
            self.show_templayout()

    def eval_net(self, net):
        insertoin_loss = (
            self.prop * net.wirelength * 1e-4 + self.cross * net.crossing_num
        )
        return insertoin_loss

    def registar_net(self, cur_net):
        if self._config.show_temp:
            for comp in cur_net.wg_component:
                self._cirdb.tempLayout.add_ref(comp)
            self._cirdb.tempLayout.show()
        self.updateHistoryMap(cur_net.origin_path, cur_net)
        self._drcmgr.updateBitmap(cur_net.rect_route, cur_net.netName)
        curnet_name = cur_net.netName
        for netID in cur_net.crossing_nets:
            net = self._cirdb.dbNets[netID]
            net.crossing_nets.add(curnet_name)

    def ripupfailedNets(self):
        if len(self.global_failedNet) == 0:
            return True
        else:
            self._ripup_times += 1
            bitmap = self._drcmgr.bitMap.bitMap
            for netID in self.global_failedNet:
                net = self._cirdb.dbNets[netID]
                for x, y in net.rwguide:
                    bitmap[x][y].delete(netID)
                net.clearNet(self._cirdb)

                if self._ripup_times == -1:
                    net.crossing_budget = max(net.crossing_budget, net.maximum_crossing)
                    net.crossing_budget = 100
                net.maximum_crossing = 0
            if self._ripup_times == 2:
                self.clearHistoryMap()
            self.global_failedNet.clear()
            return False

    def checkFailed(self):
        for _, value in self._cirdb._dbNets.items():
            if value.vionets == 0:
                value.crossing_num = len(value.crossing_nets)
        for _, value in self._cirdb._dbNets.items():
            if value.vionets != 0:
                value.crossing_num = value.vionets
                for net in value.vioNets:
                    self._cirdb._dbNets[net].crossing_num += 1

        radius = self._config.dr.bend_radius
        for net_name, current_net in self._cirdb.dbNets.items():
            segments, start_port, end_port = current_net.routed_path[0]
            align = alignment(
                segments, start_port, end_port, radius, bend=gf.path.euler
            )
            if align is not None:
                current_net.routed_path[0][0] = align

        for net_name, current_net in self._cirdb.dbNets.items():
            path_count = len(current_net.routed_path)
            crossing_number = path_count - 1
            current_net.crossing_num += crossing_number
            current_net.wirelength = 0
            for i, path in enumerate(current_net.routed_path):
                P, _, _ = smooth(
                    points=path[0],
                    radius=self._config.dr.bend_radius - 1e-9,
                    bend=gf.path.euler,
                    # bend=gf.path.arc,
                    use_eff=True,
                )
                route = gf.path.extrude(P, width=0.5, layer=(1, 0))
                # Rotate the accessing waveguide

                def generate_accessing_waveguide(route, path, index, access_waveguide):
                    port1 = path[index]
                    orient = round(port1.orientation)
                    port2 = route.ports["o1"] if index == 1 else route.ports["o2"]
                    # rotation_angle = None

                    match orient:
                        case 0 | 180:
                            waveguide_length = route_bundle_sbend(
                                access_waveguide,
                                [port1],
                                [port2],
                            )
                            return None, waveguide_length
                        case 90 | 270:
                            waveguide_length = route_bundle_sbend(
                                access_waveguide,
                                [port1],
                                [port2],
                            )
                            return None, waveguide_length
                        case 45:
                            x1, y1 = port1.dcenter
                            x2, y2 = port2.dcenter
                            dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                            port1 = gf.Port(
                                name="o1",
                                orientation=0,
                                center=(x1, y1),
                                width=0.5,
                                layer=1,
                            )
                            port2 = gf.Port(
                                name="o2",
                                orientation=180,
                                center=(x1 + dist, y1),
                                width=0.5,
                                layer=1,
                            )
                            # generate fake component
                            #
                            waveguide_length = route_bundle_sbend(
                                access_waveguide,
                                [port1],
                                [port2],
                            )
                            return (45, x1, y1), waveguide_length
                        case 135:
                            x1, y1 = port1.dcenter
                            x2, y2 = port2.dcenter
                            dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                            port1 = gf.Port(
                                name="o1",
                                orientation=180,
                                center=(x1, y1),
                                width=0.5,
                                layer=1,
                            )
                            port2 = gf.Port(
                                name="o2",
                                orientation=0,
                                center=(x1 - dist, y1),
                                width=0.5,
                                layer=1,
                            )
                            waveguide_length = route_bundle_sbend(
                                access_waveguide,
                                [port1],
                                [port2],
                            )
                            return (-45, x1, y1), waveguide_length
                        case 225:
                            x1, y1 = port1.dcenter
                            x2, y2 = port2.dcenter
                            dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                            port1 = gf.Port(
                                name="o1",
                                orientation=180,
                                center=(x1, y1),
                                width=0.5,
                                layer=1,
                            )
                            port2 = gf.Port(
                                name="o2",
                                orientation=0,
                                center=(x1 - dist, y1),
                                width=0.5,
                                layer=1,
                            )
                            waveguide_length = route_bundle_sbend(
                                access_waveguide,
                                [port1],
                                [port2],
                            )
                            return (45, x1, y1), waveguide_length
                        case 315:
                            x1, y1 = port1.dcenter
                            x2, y2 = port2.dcenter
                            dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                            port1 = gf.Port(
                                name="o1",
                                orientation=0,
                                center=(x1, y1),
                                width=0.5,
                                layer=1,
                            )
                            port2 = gf.Port(
                                name="o2",
                                orientation=180,
                                center=(x1 + dist, y1),
                                width=0.5,
                                layer=1,
                            )
                            waveguide_length = route_bundle_sbend(
                                access_waveguide,
                                [port1],
                                [port2],
                            )
                            return (-45, x1, y1), waveguide_length

                access_waveguide1 = gf.Component()
                access_waveguide2 = gf.Component()
                rotation_angle1, component_length1 = generate_accessing_waveguide(
                    route, path, index=1, access_waveguide=access_waveguide1
                )
                rotation_angle2, component_length2 = generate_accessing_waveguide(
                    route, path, index=2, access_waveguide=access_waveguide2
                )

                final_component = gf.Component(name=(current_net.netName + str(i)))
                ref1 = final_component << access_waveguide1
                ref2 = final_component << access_waveguide2
                if rotation_angle1 is not None:
                    ref1.rotate(
                        rotation_angle1[0], (rotation_angle1[1], rotation_angle1[2])
                    )
                if rotation_angle2 is not None:
                    ref2.rotate(
                        rotation_angle2[0], (rotation_angle2[1], rotation_angle2[2])
                    )
                current_net.wirelength += component_length1
                current_net.wirelength += component_length2
                current_net.wirelength += route.info.model_extra.get("length", 0)
                final_component.add_ref(route)

                components = current_net.wg_component
                components.clear()
                components.append(final_component)
                self._cirdb.layout.add_ref(final_component)

    def clearHistoryMap(self):
        self.historyMap = np.zeros(
            (self._drcmgr.bitMap_width, self._drcmgr.bitMap_height, 8), dtype=np.int32
        )

    def mark_conflict_nets(self, violatedNets, cur_net):
        self.global_failedNet.add(cur_net)
        for net in violatedNets:
            self.global_failedNet.add(net)

    def updateHistoryMap(self, path, net):
        orientations = self.orientations
        historyCost = self._config.dr.historyCost
        historyMap = self.historyMap
        ports1 = set(net.NetPort1.port_grids)
        ports2 = set(net.NetPort2.port_grids)
        for point in path:
            x, y, ori = point
            check = (x, y)
            if check in ports1 or check in ports2:
                continue
            else:
                ori_index = orientations[ori]
                historyMap[x][y][ori_index] += historyCost

    def post_processing(self):
        """post-processing after routing sucessfully"""
        radius = self._config.dr.bend_radius
        for net_name, current_net in self._cirdb.dbNets.items():
            segments, start_port, end_port = current_net.routed_path[0]
            align = alignment(
                segments, start_port, end_port, radius, bend=gf.path.euler
            )
            if align is not None:
                current_net.routed_path[0][0] = align

        for cur_net_name, current_net in self._cirdb.dbNets.items():
            for crossing_net_name in current_net.crossing_nets:
                cross_net = self._cirdb.dbNets[crossing_net_name]
                if cur_net_name in cross_net.crossing_nets:  # Double check the crossing
                    cross_net.crossing_nets.remove(cur_net_name)

                    path1 = current_net.routed_path
                    path2 = cross_net.routed_path
                    subpaths1, subpaths2, crossing = self.spilt_path(path1, path2)

                    crossing_ports = self.add_crossing_component(crossing)

                    index1, segment1_parts, ports1 = subpaths1
                    index2, segment2_parts, ports2 = subpaths2
                    subpath1 = path1[index1]
                    subpath2 = path2[index2]

                    subpath1_1 = [
                        segment1_parts[0],
                        subpath1[1],
                        crossing_ports[ports1[0]],
                    ]
                    subpath1_2 = [
                        segment1_parts[1],
                        crossing_ports[ports1[1]],
                        subpath1[2],
                    ]
                    subpath2_1 = [
                        segment2_parts[0],
                        subpath2[1],
                        crossing_ports[ports2[0]],
                    ]
                    subpath2_2 = [
                        segment2_parts[1],
                        crossing_ports[ports2[1]],
                        subpath2[2],
                    ]

                    # update the net' paths
                    path1[index1] = subpath1_1
                    path1.append(subpath1_2)
                    path2[index2] = subpath2_1
                    path2.append(subpath2_2)

        for net_name, current_net in self._cirdb.dbNets.items():
            path_count = len(current_net.routed_path)
            crossing_number = max(0, path_count - 1)
            current_net.crossing_num = crossing_number
            current_net.wirelength = 0
            for i, path in enumerate(current_net.routed_path):
                P, _, _ = smooth(
                    points=path[0],
                    radius=self._config.dr.bend_radius - 1e-9,
                    bend=gf.path.euler,
                    # bend=gf.path.arc,
                    use_eff=True,
                )
                route = gf.path.extrude(P, width=0.5, layer=(1, 0))

                def generate_accessing_waveguide(route, path, index, access_waveguide):
                    port1 = path[index]
                    orient = round(port1.orientation)
                    port2 = route.ports["o1"] if index == 1 else route.ports["o2"]
                    # rotation_angle = None

                    match orient:
                        case 0 | 180:
                            waveguide_length = route_bundle_sbend(
                                access_waveguide,
                                [port1],
                                [port2],
                            )
                            return None, waveguide_length
                        case 90 | 270:
                            waveguide_length = route_bundle_sbend(
                                access_waveguide,
                                [port1],
                                [port2],
                            )
                            return None, waveguide_length
                        case 45:
                            x1, y1 = port1.dcenter
                            x2, y2 = port2.dcenter
                            dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                            port1 = gf.Port(
                                name="o1",
                                orientation=0,
                                center=(x1, y1),
                                width=0.5,
                                layer=1,
                            )
                            port2 = gf.Port(
                                name="o2",
                                orientation=180,
                                center=(x1 + dist, y1),
                                width=0.5,
                                layer=1,
                            )
                            # generate fake component
                            waveguide_length = route_bundle_sbend(
                                access_waveguide,
                                [port1],
                                [port2],
                            )
                            return (45, x1, y1), waveguide_length
                        case 135:
                            x1, y1 = port1.dcenter
                            x2, y2 = port2.dcenter
                            dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                            port1 = gf.Port(
                                name="o1",
                                orientation=180,
                                center=(x1, y1),
                                width=0.5,
                                layer=1,
                            )
                            port2 = gf.Port(
                                name="o2",
                                orientation=0,
                                center=(x1 - dist, y1),
                                width=0.5,
                                layer=1,
                            )
                            waveguide_length = route_bundle_sbend(
                                access_waveguide,
                                [port1],
                                [port2],
                            )
                            return (-45, x1, y1), waveguide_length
                        case 225:
                            x1, y1 = port1.dcenter
                            x2, y2 = port2.dcenter
                            dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                            port1 = gf.Port(
                                name="o1",
                                orientation=180,
                                center=(x1, y1),
                                width=0.5,
                                layer=1,
                            )
                            port2 = gf.Port(
                                name="o2",
                                orientation=0,
                                center=(x1 - dist, y1),
                                width=0.5,
                                layer=1,
                            )
                            waveguide_length = route_bundle_sbend(
                                access_waveguide,
                                [port1],
                                [port2],
                            )
                            return (45, x1, y1), waveguide_length
                        case 315:
                            x1, y1 = port1.dcenter
                            x2, y2 = port2.dcenter
                            dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                            port1 = gf.Port(
                                name="o1",
                                orientation=0,
                                center=(x1, y1),
                                width=0.5,
                                layer=1,
                                # cross_section="xs"
                            )
                            port2 = gf.Port(
                                name="o2",
                                orientation=180,
                                center=(x1 + dist, y1),
                                width=0.5,
                                layer=1,
                                # cross_section="xs"
                            )
                            waveguide_length = route_bundle_sbend(
                                access_waveguide,
                                [port1],
                                [port2],
                            )
                            return (-45, x1, y1), waveguide_length

                access_waveguide1 = gf.Component()
                access_waveguide2 = gf.Component()
                rotation_angle1, component_length1 = generate_accessing_waveguide(
                    route, path, index=1, access_waveguide=access_waveguide1
                )
                rotation_angle2, component_length2 = generate_accessing_waveguide(
                    route, path, index=2, access_waveguide=access_waveguide2
                )

                final_component = gf.Component(name=(current_net.netName + str(i)))
                ref1 = final_component << access_waveguide1
                ref2 = final_component << access_waveguide2
                if rotation_angle1 is not None:
                    ref1.rotate(
                        rotation_angle1[0], (rotation_angle1[1], rotation_angle1[2])
                    )
                if rotation_angle2 is not None:
                    ref2.rotate(
                        rotation_angle2[0], (rotation_angle2[1], rotation_angle2[2])
                    )
                current_net.wirelength += component_length1
                current_net.wirelength += component_length2
                current_net.wirelength += route.info.model_extra.get("length", 0)
                final_component.add_ref(route)

                components = current_net.wg_component
                components.clear()
                components.append(final_component)
                self._cirdb.layout.add_ref(final_component)

        """
            1. Align the segments
            2. Spilit the segments 
            3. Add the crossing
        """

    def evaluation(self):
        import networkx as nx

        propagation_loss = self.prop * 1e-4
        bending_loss = 0.01 / 90
        crossing_loss = self.cross

        dbNets = self._cirdb.dbNets
        DRV = 0
        if self._config.eval == "comp":
            for net_name, current_net in dbNets.items():
                inst = current_net.NetPort1.instanceName
                device_loss = self._cirdb.device_loss[inst]
                output_node = current_net.NetPort2.idBlk
                if self._cirdb._netlist_graph.out_degree(output_node) == 0:
                    output_inst = current_net.NetPort2.instanceName
                    output_device_loss = self._cirdb.device_loss[output_inst]
                    device_loss += output_device_loss

                insertion_loss = (
                    current_net.wirelength * propagation_loss
                    + current_net.bending * bending_loss
                    + current_net.crossing_num * crossing_loss
                ) + device_loss
                DRV += current_net.vionets
                print(
                    f"Net: {net_name}, WL: {current_net.wirelength:.3f} um, Accumulated bend: {current_net.bending:.1f},Crossing: {current_net.crossing_num}, insertion loss: {insertion_loss}, DRV: {current_net.vionets}"
                )
                self._cirdb._netlist_graph.add_edge(
                    current_net.NetPort1.idBlk,
                    current_net.NetPort2.idBlk,
                    weight=insertion_loss,
                    label=current_net.netName,
                )

            criticalPath = nx.dag_longest_path(self._cirdb._netlist_graph)
            lenCriticalPath = nx.dag_longest_path_length(self._cirdb._netlist_graph)
            node_num = len(criticalPath)
            nets = []
            crossing = 0
            WL = 0
            DRV_path = 0
            for i, value in enumerate(criticalPath):
                if i < node_num - 1:
                    edge = self._cirdb._netlist_graph[value][criticalPath[i + 1]]
                    net = edge["label"]
                    crossing += dbNets[net].crossing_num
                    WL += dbNets[net].wirelength
                    DRV_path += dbNets[net].vionets
                    nets.append(net)
            print(
                f"criticalPath: {nets}, WL: {WL:.2f}, crossing: {crossing}, Max Insertion loss: {lenCriticalPath:.3f}, DRV: {DRV}, DRV_path: {DRV_path}"
            )
        else:
            i = 0
            ilmax1 = 0
            path1 = (0, 0, 0)
            path2 = (0, 0, 0)
            ilmax2 = 0
            DRV = 0
            for net_name, current_net in dbNets.items():
                i += 1
                insertion_loss = (
                    current_net.wirelength * propagation_loss
                    + current_net.bending * bending_loss
                    + current_net.crossing_num * crossing_loss
                )
                DRV += current_net.vionets
                print(
                    f"Net: {net_name}, WL: {current_net.wirelength:.3f} um, Accumulated bend: {current_net.bending:.1f}, Crossing: {current_net.crossing_num}, insertion loss: {insertion_loss}, DRV: {current_net.vionets}"
                )
                if i <= 8 and insertion_loss > ilmax1:
                    ilmax1 = insertion_loss
                    path1 = (
                        insertion_loss,
                        current_net.crossing_num,
                        current_net.wirelength,
                    )
                elif i > 8 and insertion_loss > ilmax2:
                    ilmax2 = insertion_loss
                    path2 = (
                        insertion_loss,
                        current_net.crossing_num,
                        current_net.wirelength,
                    )
            ilmax = path1[0] + path2[0] + crossing_loss * 6
            CR = path1[1] + path2[1] + 6
            WL = path1[2] + path2[2]
            print(f"criticalPath: ilmax = {ilmax}, CR = {CR}, WL = {WL}, DRV = {DRV}")

    def add_crossing_component(self, crossing):
        point, ori = crossing
        c = gf.Component()
        ref1 = c << gf.components.crossing()
        ref1.rotate(ori)
        ref1.move(point)
        c.add_ports(ref1.ports)
        self._cirdb.layout.add_ref(c)

        return c.ports

    def spilt_path(self, net1_path, net2_path):
        for index1, subpath1 in enumerate(net1_path):
            segment1 = LineString(subpath1[0])
            for index2, subpath2 in enumerate(
                net2_path
            ):  # Check crossing for each segment
                segment2 = LineString(subpath2[0])
                intersection = segment1.intersection(segment2)
                if not intersection.is_empty and isinstance(
                    intersection, Point
                ):  # Only one time crossing is allowed
                    distance_to_intersection1 = segment1.project(intersection)
                    distance_to_intersection2 = segment2.project(intersection)
                    segment1_parts, ports1, ori = self.spilt_line_string(
                        segment1, distance_to_intersection1
                    )
                    segment2_parts, ports2, _ = self.spilt_line_string(
                        segment2, distance_to_intersection2
                    )
                    return (
                        [index1, segment1_parts, ports1],
                        [index2, segment2_parts, ports2],
                        [intersection.coords._coords[0], ori],
                    )

    def spilt_line_string(self, line, distance):
        if distance <= 0.0 or distance >= line.length:
            return [LineString(line)]
        coords = list(line.coords)
        CL_0 = 4.5
        CL_45 = 4.5
        for i, p in enumerate(coords):
            pd = line.project(Point(p))
            if pd > distance:
                cp = line.interpolate(distance)
                x, y = coords[i]
                dx = cp.x - x
                dy = cp.y - y
                if dx == 0:
                    slope = 90
                else:
                    slope = dy / dx
                if slope != float("inf") and slope != float("-inf"):
                    slope = round(slope)
                else:
                    slope = 90
                match slope:
                    case 0:
                        if dx < 0:
                            return (
                                [
                                    coords[:i] + [(cp.x - CL_0, cp.y)],
                                    [(cp.x + CL_0, cp.y)] + coords[i:],
                                ],
                                ("o1", "o3"),
                                0,
                            )
                        else:
                            return (
                                [
                                    coords[:i] + [(cp.x + CL_0, cp.y)],
                                    [(cp.x - CL_0, cp.y)] + coords[i:],
                                ],
                                ("o3", "o1"),
                                0,
                            )
                    case 90:
                        if dy < 0:
                            return (
                                [
                                    coords[:i] + [(cp.x, cp.y - CL_0)],
                                    [(cp.x, cp.y + CL_0)] + coords[i:],
                                ],
                                ("o4", "o2"),
                                0,
                            )
                        else:
                            return (
                                [
                                    coords[:i] + [(cp.x, cp.y + CL_0)],
                                    [(cp.x, cp.y - CL_0)] + coords[i:],
                                ],
                                ("o2", "o4"),
                                0,
                            )
                    case -1:
                        if dy > 0:
                            return (
                                [
                                    coords[:i] + [(cp.x - CL_45, cp.y + CL_45)],
                                    [(cp.x + CL_45, cp.y - CL_45)] + coords[i:],
                                ],
                                ("o1", "o3"),
                                -45,
                            )
                        else:
                            return (
                                [
                                    coords[:i] + [(cp.x + CL_45, cp.y - CL_45)],
                                    [(cp.x - CL_45, cp.y + CL_45)] + coords[i:],
                                ],
                                ("o3", "o1"),
                                -45,
                            )
                    case 1:
                        if dy > 0:
                            return (
                                [
                                    coords[:i] + [(cp.x + CL_45, cp.y + CL_45)],
                                    [(cp.x - CL_45, cp.y - CL_45)] + coords[i:],
                                ],
                                ("o2", "o4"),
                                -45,
                            )
                        else:
                            return (
                                [
                                    coords[:i] + [(cp.x - CL_45, cp.y - CL_45)],
                                    [(cp.x + CL_45, cp.y + CL_45)] + coords[i:],
                                ],
                                ("o4", "o2"),
                                -45,
                            )
