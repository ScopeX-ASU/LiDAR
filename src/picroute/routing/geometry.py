"""PDK-aware conversion of LiDAR backbones into GDSFactory geometry."""

from __future__ import annotations

from functools import partial

import gdsfactory as gf
import numpy as np
from kfactory.routing.aa.optical import route as route_all_angle


class RouteGeometry:
    """Materialize LiDAR paths using one configured optical cross-section."""

    def __init__(self, config) -> None:
        self.cross_section_spec = config.dr.get("cross_section", "strip")
        self.cross_section = gf.get_cross_section(self.cross_section_spec)
        self.radius = float(config.dr.bend_radius)

    @property
    def width(self) -> float:
        return float(self.cross_section.width)

    @property
    def layer(self):
        return self.cross_section.layer

    def smooth(self, points):
        return gf.path.smooth(
            points=points,
            radius=self.radius - 1e-9,
            bend=gf.path.euler,
            use_eff=True,
        )

    def extrude(self, path, *, width: float | None = None) -> gf.Component:
        if width is None:
            return gf.path.extrude(path, cross_section=self.cross_section_spec)
        return gf.path.extrude(path, width=width, layer=self.layer)

    def occupied_grid_cells(self, component: gf.Component, bitmap) -> set[tuple[int, int]]:
        """Return grid cells intersecting the routed layer's filled KLayout region."""
        region = component.get_region(self.layer, merge=True)
        if region.is_empty():
            return set()

        dbu = component.kcl.dbu
        bbox = region.bbox()
        lower = bitmap.to_grid([bbox.left * dbu, bbox.bottom * dbu])
        upper = bitmap.to_grid([bbox.right * dbu, bbox.top * dbu])
        cells = set()
        for x in range(max(0, int(lower[0])), min(bitmap.width - 1, int(upper[0])) + 1):
            for y in range(
                max(0, int(lower[1])), min(bitmap.height - 1, int(upper[1])) + 1
            ):
                world_min = bitmap.to_world([x, y], center=False)
                world_max = world_min + bitmap._resolution
                box = gf.kdb.Box(
                    round(world_min[0] / dbu),
                    round(world_min[1] / dbu),
                    round(world_max[0] / dbu),
                    round(world_max[1] / dbu),
                )
                if not (region & gf.kdb.Region(box)).is_empty():
                    cells.add((x, y))
        return cells

    def place_all_angle(self, points) -> gf.Component:
        """Place an instance-based kfactory route along an exact LiDAR backbone."""
        component = gf.Component()
        route = route_all_angle(
            component,
            width=self.width,
            backbone=[gf.kdb.DPoint(*point) for point in points],
            straight_factory=partial(
                gf.components.straight_all_angle,
                cross_section=self.cross_section_spec,
            ),
            bend_factory=partial(
                gf.components.bend_euler_all_angle,
                radius=self.radius,
                cross_section=self.cross_section_spec,
            ),
        )
        for name, port in (("o1", route.start_port), ("o2", route.end_port)):
            component.add_port(
                name=name,
                center=port.dcenter,
                width=port.dwidth,
                orientation=port.orientation,
                layer=port.layer,
            )
        component.insert_vinsts()
        return component

    def _access_waveguide(
        self,
        route: gf.Component,
        endpoint,
        index: int,
    ) -> tuple[gf.Component, tuple[float, float, float] | None, float]:
        access_waveguide = gf.Component()
        port1 = endpoint
        orientation = round(port1.orientation) % 360
        port2 = route.ports["o1"] if index == 1 else route.ports["o2"]
        rotation = None

        if orientation not in {0, 90, 180, 270}:
            x1, y1 = port1.dcenter
            x2, y2 = port2.dcenter
            distance = float(np.hypot(x2 - x1, y2 - y1))
            if orientation in {45, 315}:
                port1_orientation, port2_orientation = 0, 180
                direction = 1
            elif orientation in {135, 225}:
                port1_orientation, port2_orientation = 180, 0
                direction = -1
            else:
                raise ValueError(f"Unsupported access-port orientation {orientation}")
            port1 = gf.Port(
                name="o1",
                orientation=port1_orientation,
                center=(x1, y1),
                width=self.width,
                layer=self.layer,
            )
            port2 = gf.Port(
                name="o2",
                orientation=port2_orientation,
                center=(x1 + direction * distance, y1),
                width=self.width,
                layer=self.layer,
            )
            rotation_angle = 45 if orientation in {45, 225} else -45
            rotation = (rotation_angle, float(x1), float(y1))

        routes = gf.routing.route_bundle(
            access_waveguide,
            [port1],
            [port2],
            cross_section=self.cross_section_spec,
            radius=self.radius,
            sbend="bend_s",
            on_collision=None,
            on_placer_error="error",
        )
        length = sum(
            float(instance.cell.info.get("length", 0))
            for instance in routes[0].instances
        )
        return access_waveguide, rotation, length

    def materialize_record(self, path_record) -> tuple[gf.Component, float]:
        """Create one routed component and return it with its physical length."""
        path = self.smooth(path_record[0])
        route_shape = self.extrude(path)
        access1, rotation1, length1 = self._access_waveguide(
            route_shape, path_record[1], index=1
        )
        access2, rotation2, length2 = self._access_waveguide(
            route_shape, path_record[2], index=2
        )
        route = self.place_all_angle(path_record[0])

        component = gf.Component()
        ref1 = component << access1
        ref2 = component << access2
        if rotation1 is not None:
            ref1.drotate(rotation1[0], (rotation1[1], rotation1[2]))
        if rotation2 is not None:
            ref2.drotate(rotation2[0], (rotation2[1], rotation2[2]))
        component.add_ref(route)
        return component, length1 + length2 + float(path.length())
