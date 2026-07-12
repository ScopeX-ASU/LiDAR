"""Compact directional occupancy grid used by LiDAR's A* router."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

EMPTY = -1
BLOCKAGE = -2
PORT = -3
COMPOUND = -4

_TYPE_TO_CODE = {
    "empty": EMPTY,
    "blk": BLOCKAGE,
    "port": PORT,
    "compound": COMPOUND,
}
_CODE_TO_TYPE = {value: key for key, value in _TYPE_TO_CODE.items()}


class bitmapNode:
    """Lightweight view of one cell in a :class:`Bitmap`."""

    __slots__ = ("_grid", "_x", "_y")

    def __init__(self, grid: Bitmap, x: int, y: int) -> None:
        self._grid = grid
        self._x = int(x)
        self._y = int(y)

    @property
    def _key(self) -> tuple[int, int]:
        return self._x, self._y

    @property
    def type(self):
        code = int(self._grid._types[self._x, self._y])
        return _CODE_TO_TYPE.get(code, code)

    @property
    def nodetype(self):
        return self.type

    @property
    def blkID(self):
        return self._grid._owners.get(self._key)

    @property
    def netID(self) -> list[Any]:
        return list(self._grid._routes.get(self._key, {}))

    @property
    def _wgtype(self) -> dict[Any, int]:
        return self._grid._routes.get(self._key, {})

    @property
    def length(self) -> int:
        return len(self._grid._routes.get(self._key, {}))

    def update(self, nodetype: str, id=0, wgtype=None) -> None:
        if nodetype not in {"empty", "blk", "waveguide", "port"}:
            raise ValueError(f"Unsupported bitmap node type {nodetype!r}")

        current_type = self.type
        if nodetype == "waveguide":
            if current_type in {"blk", "port"}:
                return
            routes = self._grid._routes.setdefault(self._key, {})
            routes[id] = wgtype
            self._grid._types[self._x, self._y] = (
                COMPOUND if len(routes) > 1 else int(wgtype)
            )
        elif nodetype == "blk":
            self._grid._types[self._x, self._y] = BLOCKAGE
            self._grid._owners[self._key] = id
            self._grid._routes.pop(self._key, None)
        elif nodetype == "port":
            if current_type == "empty":
                self._grid._types[self._x, self._y] = PORT
                self._grid._owners[self._key] = id
        elif current_type not in {"blk", "port"}:
            self._grid._types[self._x, self._y] = EMPTY
            self._grid._routes.pop(self._key, None)

    def delete(self, net_name) -> None:
        if self.type in {"blk", "port", "empty"}:
            return
        routes = self._grid._routes.get(self._key)
        if not routes:
            self._grid._types[self._x, self._y] = EMPTY
            return
        routes.pop(net_name, None)
        if not routes:
            self._grid._routes.pop(self._key, None)
            self._grid._types[self._x, self._y] = EMPTY
        elif len(routes) == 1:
            self._grid._types[self._x, self._y] = next(iter(routes.values()))
        else:
            self._grid._types[self._x, self._y] = COMPOUND

    def __str__(self) -> str:
        return str(self.type)


class _BitmapRow:
    __slots__ = ("_grid", "_x")

    def __init__(self, grid: Bitmap, x: int) -> None:
        self._grid = grid
        self._x = int(x)

    def __getitem__(self, y: int) -> bitmapNode:
        return self._grid.node(self._x, y)


class _BitmapView:
    __slots__ = ("_grid",)

    def __init__(self, grid: Bitmap) -> None:
        self._grid = grid

    def __getitem__(self, index):
        if isinstance(index, tuple):
            return self._grid.node(index[0], index[1])
        return _BitmapRow(self._grid, index)

    @property
    def shape(self) -> tuple[int, int]:
        return self._grid.width, self._grid.height


class Bitmap:
    def __init__(
        self,
        die_size: Any,
        resolution: float = 1,
        distance: int = 1,
    ) -> None:
        self.origin = np.asarray(die_size[0], dtype=float)
        upper = np.asarray(die_size[1], dtype=float)
        self._resolution = float(resolution)
        self.width = math.ceil((upper[0] - self.origin[0]) / self._resolution)
        self.height = math.ceil((upper[1] - self.origin[1]) / self._resolution)
        self._distance = float(distance)
        self._types = np.full((self.width, self.height), EMPTY, dtype=np.int16)
        self._owners: dict[tuple[int, int], Any] = {}
        self._routes: dict[tuple[int, int], dict[Any, int]] = {}
        self._view = _BitmapView(self)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def node(self, x: int, y: int) -> bitmapNode:
        x, y = int(x), int(y)
        if not self.in_bounds(x, y):
            raise IndexError(f"Bitmap index {(x, y)} outside {(self.width, self.height)}")
        return bitmapNode(self, x, y)

    def to_grid(self, point) -> np.ndarray:
        return np.floor(
            (np.asarray(point, dtype=float) - self.origin) / self._resolution
        ).astype(np.int32)

    def to_world(self, point, *, center: bool = True) -> np.ndarray:
        offset = self._resolution / 2 if center else 0
        return self.origin + np.asarray(point, dtype=float) * self._resolution + offset

    def initMap(self, blockages) -> None:
        for blockage in blockages:
            lower = self.to_grid(
                [
                    blockage.bbox[0] - self._distance,
                    blockage.bbox[1] - self._distance,
                ]
            )
            upper = self.to_grid(
                [
                    blockage.bbox[2] + self._distance,
                    blockage.bbox[3] + self._distance,
                ]
            )
            xmin, ymin = np.maximum(lower, 0)
            xmax, ymax = np.minimum(upper, [self.width - 1, self.height - 1])
            for x in range(int(xmin), int(xmax) + 1):
                for y in range(int(ymin), int(ymax) + 1):
                    self.node(x, y).update("blk", blockage.instanceName)

    def _is_empty(self, location: np.ndarray) -> bool:
        x, y = map(int, location)
        return self.in_bounds(x, y) and self.node(x, y).type == "empty"

    def _reserve_port(self, port, net_name: str, port_length: int) -> None:
        orientations = {
            0.0: np.array([1, 0]),
            90.0: np.array([0, 1]),
            180.0: np.array([-1, 0]),
            270.0: np.array([0, -1]),
        }
        step = orientations[float(port.orientation)]
        location = self.to_grid(port.center)
        while self.in_bounds(*location) and self.node(*location).type == "blk":
            location += step
        if not self.in_bounds(*location):
            return

        perpendicular = np.array([step[1], step[0]])
        positive = location + perpendicular
        negative = location - perpendicular
        if not self._is_empty(positive) and self._is_empty(negative):
            location = negative
        elif not self._is_empty(negative) and self._is_empty(positive):
            location = positive

        for _ in range(port_length):
            if not self.in_bounds(*location):
                break
            self.node(*location).update("port", net_name)
            port.port_grids.append(tuple(map(int, location)))
            location += step

    def initPorts(self, dbNets, port_length=5) -> None:
        for net_name, net in dbNets.items():
            self._reserve_port(net.NetPort1, net_name, port_length)
            self._reserve_port(net.NetPort2, net_name, port_length)

    @property
    def bitMap(self) -> _BitmapView:
        return self._view
