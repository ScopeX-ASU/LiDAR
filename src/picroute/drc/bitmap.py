"""
Date: 2024-06-05 00:15:56
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-02-05 01:04:56
FilePath: /PICRoute/src/picroute/drc/bitmap.py
"""

from typing import Any

import numpy as np


class bitmapNode:
    """Support empty, blockage, waveguide, compound(multiple waveguide in one grid) and port type now.
    blkID: record the blockage instance ID when type is blk or the netID when type is port type.
    self.netID: a list that record the nets the grid containing
    self._wgtype: a list that record the wgtype the grid containing
    self.length: length of self.netID

    """

    __slots__ = ["nodetype", "blkID", "_wgtype", "length", "_netID"]

    def __init__(self, nodetype: str = "empty"):
        self.nodetype = nodetype

    @property
    def netID(self):
        if not hasattr(self, "_netID"):
            assert isinstance(self.nodetype, str) and self.nodetype in {
                "empty",
                "blk",
                "waveguide",
                "compound",
                "port",
            }
            if hasattr(self, "blkID"):
                pass
            else:
                self.blkID = None
            self._netID = []  # netID
            self._wgtype = {}  # 1: bend, 45, 90, 135, 180
            self.length = 0
        return self._netID

    def update(self, nodetype, id=0, wgtype=None):
        assert type(nodetype) == str and nodetype in {
            "empty",
            "blk",
            "waveguide",
            "port",
        }
        if nodetype == "waveguide":
            # assert wgtype is not None and wgtype in {1, 45, 90, 135, 180}
            if self.type == "blk" or self.type == "port":
                return
            self.netID.append(id)
            self._wgtype[id] = wgtype
            self.length += 1
            if self.length > 1:
                self.nodetype = "compound"
            else:
                self.nodetype = wgtype  # str / int
        elif nodetype == "blk":
            self.nodetype = nodetype
            self.blkID = id
        elif nodetype == "empty" and self.nodetype != "blk" and self.nodetype != "port":
            self.nodetype = nodetype
            self._netID = []
            self._wgtype = {}
            self.length = 0

        elif nodetype == "port":
            if self.nodetype == "empty":
                self.nodetype = nodetype
                self.blkID = id

    def delete(self, netName):
        match self.nodetype:
            case "blk":
                pass
            case "port":
                pass
            case "compound":
                del self._wgtype[netName]
                self.length -= 1
                if self.length == 1:
                    for netID, value in self._wgtype.items():
                        self.nodetype = value
                        self._netID = [netID]
            case _:
                del self._wgtype[netName]
                self.length = 0
                self.nodetype = "empty"
                self._netID = []

    @property
    def type(self):
        return self.nodetype

    def __str__(self) -> str:
        return self.type


class Bitmap:
    def __init__(
        self,
        die_size: Any,
        resolution: float = 1,
        distance: int = 1,
    ) -> None:
        self.width = round((die_size[1][0] - die_size[0][0]) / resolution)
        self.height = round((die_size[1][1] - die_size[0][1]) / resolution)
        self._distance = distance
        self._resolution = resolution
        self._bitmap = np.empty((self.width, self.height), dtype=object)
        self._bitmap = np.vectorize(lambda _: bitmapNode())(self._bitmap)

    def initMap(self, blockages):
        for blk in blockages:
            xmin = int(abs(blk.bbox[0] - self._distance) / self._resolution)
            xmax = int(abs(blk.bbox[2] + self._distance) / self._resolution)
            ymin = int(abs(blk.bbox[1] - self._distance) / self._resolution)
            ymax = int(abs(blk.bbox[3] + self._distance) / self._resolution)
            for x in range(xmin, xmax + 1):
                for y in range(ymin, ymax + 1):
                    try:
                        self._bitmap[x][y].update("blk", blk.instanceName)
                    except:
                        pass

    def initPorts(self, dbNets, port_length=5):
        orientations = {
            0.0: np.array([1, 0]),
            90.0: np.array([0, 1]),
            180.0: np.array([-1, 0]),
            270.0: np.array([0, -1]),
        }

        for net_name, net in dbNets.items():
            Port1 = net.NetPort1
            Port2 = net.NetPort2

            ori = orientations[Port1.port.orientation]
            loc = (np.array(Port1.port.center) / self._resolution).astype(np.int32)

            while self._bitmap[loc[0]][loc[1]].type == "blk":
                loc += ori
            if Port1.port.orientation == 0 or Port1.port.orientation == 180:
                if (
                    self._bitmap[loc[0]][loc[1] + 1].type == "empty"
                    and self._bitmap[loc[0]][loc[1] - 1].type == "empty"
                ):
                    pass
                elif (
                    self._bitmap[loc[0]][loc[1] + 1].type != "empty"
                    and self._bitmap[loc[0]][loc[1] - 1].type != "empty"
                ):
                    assert 0
                elif self._bitmap[loc[0]][loc[1] + 1].type != "empty":
                    loc += np.array([0, -1])
                else:
                    loc += np.array([0, 1])
            else:
                if (
                    self._bitmap[loc[0] + 1][loc[1]].type == "empty"
                    and self._bitmap[loc[0] - 1][loc[1]].type == "empty"
                ):
                    pass
                elif (
                    self._bitmap[loc[0] + 1][loc[1]].type != "empty"
                    and self._bitmap[loc[0] - 1][loc[1]].type != "empty"
                ):
                    assert 0
                elif self._bitmap[loc[0] + 1][loc[1]].type != "empty":
                    loc += np.array([-1, 0])
                else:
                    loc += np.array([1, 0])
            for _ in range(port_length):
                self._bitmap[loc[0]][loc[1]].update("port", net_name)
                Port1.port_grids.append((loc[0], loc[1]))
                loc += ori

            ori = orientations[Port2.port.orientation]
            loc = (np.array(Port2.port.center) / self._resolution).astype(np.int32)
            while self._bitmap[loc[0]][loc[1]].type == "blk":
                loc += ori
            # only support 0/90/180/270 orientaion now
            if Port2.port.orientation == 0 or Port2.port.orientation == 180:
                if (
                    self._bitmap[loc[0]][loc[1] + 1].type == "empty"
                    and self._bitmap[loc[0]][loc[1] - 1].type == "empty"
                ):
                    pass
                elif (
                    self._bitmap[loc[0]][loc[1] + 1].type != "empty"
                    and self._bitmap[loc[0]][loc[1] - 1].type != "empty"
                ):
                    assert 0
                elif self._bitmap[loc[0]][loc[1] + 1].type != "empty":
                    loc += np.array([0, -1])
                else:
                    loc += np.array([0, 1])
            else:
                if (
                    self._bitmap[loc[0] + 1][loc[1]].type == "empty"
                    and self._bitmap[loc[0] - 1][loc[1]].type == "empty"
                ):
                    pass
                elif (
                    self._bitmap[loc[0] + 1][loc[1]].type != "empty"
                    and self._bitmap[loc[0] - 1][loc[1]].type != "empty"
                ):
                    assert 0
                elif self._bitmap[loc[0] + 1][loc[1]].type != "empty":
                    loc += np.array([-1, 0])
                else:
                    loc += np.array([1, 0])
            for _ in range(port_length):
                self._bitmap[loc[0]][loc[1]].update("port", net_name)
                Port2.port_grids.append((loc[0], loc[1]))
                loc += ori

    @property
    def bitMap(self):
        return self._bitmap
