import copy
from collections.abc import Callable

import numpy as np
from gdsfactory.path import Path, euler
from gdsfactory.typings import Coordinates

PathFactory = Callable[..., Path]


def _compute_segments(points):
    points = np.asfarray(points)
    normals = np.diff(points, axis=0)
    normals = (normals.T / np.linalg.norm(normals, axis=1)).T
    dx = np.diff(points[:, 0])
    dy = np.diff(points[:, 1])
    ds = np.sqrt(dx**2 + dy**2)
    theta = np.degrees(np.arctan2(dy, dx))
    dtheta = np.diff(theta)
    dtheta = dtheta - 360 * np.floor((dtheta + 180) / 360)
    return points, normals, ds, theta, dtheta


def smooth(
    points: Coordinates,
    radius: float = 5.0,
    alignment: bool = False,
    start_ori: int = 0,
    start_loc: tuple = (0, 0),
    end_ori: int = 0,
    end_loc: tuple = (0, 0),
    bend: PathFactory = euler,
    **kwargs,
) -> Path:
    """Returns a smooth Path from a series of waypoints.

    Args:
        points: array-like[N][2] List of waypoints for the path to follow.
        radius: radius of curvature, passed to `bend`.
        bend: bend function that returns a path that round corners.
        kwargs: Extra keyword arguments that will be passed to `bend`.

    .. plot::
        :include-source:

        import gdsfactory as gf

        p = gf.path.smooth(([0, 0], [0, 10], [10, 10]))
        p.plot()

    """
    if isinstance(points, Path):
        points = points.points

    cp = copy.deepcopy(points)
    if alignment == True:
        try:
            points, normals, ds, theta, dtheta = _compute_segments(points)
            colinear_elements = np.concatenate(
                [[False], np.abs(dtheta) < 1e-6, [False]]
            )
            if np.any(colinear_elements):
                new_points = points[~colinear_elements, :]
                points, normals, ds, theta, dtheta = _compute_segments(new_points)

            if start_ori > 180:
                start_ori -= 360
            if end_ori > 180:
                end_ori -= 360

            x1, y1 = start_loc
            x2, y2 = end_loc  #### dx1, dy1

            dx1 = x1 - points[0, 0]
            dy1 = y1 - points[0, 1]
            dx2 = x2 - points[-1, 0]
            dy2 = y2 - points[-1, 1]

            i = 0
            match start_ori:
                case 0:
                    points[i, 1] = y1
                    points[i + 1, 1] = y1
                    if len(dtheta) > 0:
                        if abs(dtheta[0]) == 45:
                            points[1, 0] += dy1
                        if dtheta[0] == -45:
                            points[1, 0] -= dy1
                        else:
                            len_dtheta = len(dtheta)
                            while (
                                len_dtheta > 2
                                and i < len_dtheta - 1
                                and abs(dtheta[i]) == 90
                                and abs(dtheta[i + 1]) == 90
                                and ds[i + 1] - dy1 < 2 * radius
                            ):
                                i += 2
                                points[i, 1] += dy1
                                points[i + 1, 1] += dy1
                            if (
                                i == len_dtheta - 1
                                and abs(dtheta[-1]) == 90
                                and ds[i + 1] - dy1 < radius
                            ):
                                points[-1, 1] += dy1
                case 90:
                    points[0, 0] = x1
                    points[1, 0] = x1
                    if len(dtheta) > 0:
                        if dtheta[0] == 45:  # 135
                            points[1, 1] -= dx1
                        elif dtheta[0] == -45:  # 45
                            points[1, 1] += dx1
                        else:
                            len_dtheta = len(dtheta)
                            while (
                                len_dtheta > 2
                                and i < len_dtheta - 1
                                and abs(dtheta[i]) == 90
                                and abs(dtheta[i + 1]) == 90
                                and ds[i + 1] - dx1 < 2 * radius
                            ):
                                i += 2
                                points[i, 0] += dx1
                                points[i + 1, 0] += dx1
                            if (
                                i == len_dtheta - 1
                                and abs(dtheta[-1]) == 90
                                and ds[i + 1] - dx1 < radius
                            ):
                                points[-1, 0] += dx1
                case 180:
                    points[0, 1] += dy1
                    points[1, 1] += dy1
                    if len(dtheta) > 0:
                        if dtheta[0] == 45:  # 45
                            points[1, 0] += dy1
                        elif dtheta[0] == -45:  # 135
                            points[1, 0] -= dy1
                        else:
                            len_dtheta = len(dtheta)
                            while (
                                len_dtheta > 2
                                and i < len_dtheta - 1
                                and abs(dtheta[i]) == 90
                                and abs(dtheta[i + 1]) == 90
                                and ds[i + 1] - dy1 < 2 * radius
                            ):
                                i += 2
                                points[i, 1] += dy1
                                points[i + 1, 1] += dy1
                            if (
                                i == len_dtheta - 1
                                and abs(dtheta[-1]) == 90
                                and ds[i + 1] - dy1 < radius
                            ):
                                points[-1, 1] += dy1
                case -90:
                    points[0, 0] = x1
                    points[1, 0] = x1
                    if len(dtheta) > 0:
                        if dtheta[0] == 45:  # 135
                            points[1, 1] -= dx1
                        elif dtheta[0] == -45:  # 45
                            points[1, 1] += dx1
                        else:
                            len_dtheta = len(dtheta)
                            while (
                                len_dtheta > 2
                                and i < len_dtheta - 1
                                and abs(dtheta[i]) == 90
                                and abs(dtheta[i + 1]) == 90
                                and ds[i + 1] - dx1 < 2 * radius
                            ):
                                i += 2
                                points[i, 0] += dx1
                                points[i + 1, 0] += dx1
                            if (
                                i == len_dtheta - 1
                                and abs(dtheta[-1]) == 90
                                and ds[i + 1] - dx1 < radius
                            ):
                                points[-1, 0] += dx1

            if len(theta) > 1 and end_ori == theta[-1]:
                match end_ori:
                    case 0:
                        if abs(dy2) < 2:
                            points[-1, 1] = y2
                            points[-2, 1] = y2
                            if dtheta[-1] == 45:
                                points[-2, 0] -= dy2
                            elif dtheta[-1] == -45:
                                points[-2, 0] += dy2
                    case 90:
                        if abs(dx2) < 2:
                            points[-1, 0] = x2
                            points[-2, 0] = x2
                            if dtheta[-1] == 45:
                                points[-2, 1] += dy2
                            elif dtheta[-1] == -45:
                                points[-2, 1] -= dy2
                    case 180:
                        if abs(dy2) < 2:
                            points[-1, 1] = y2
                            points[-2, 1] = y2
                            if dtheta[-1] == 45:
                                points[-2, 0] -= dy2
                            elif dtheta[-1] == -45:
                                points[-2, 0] += dy2
                    case -90:
                        if abs(dx2) < 2:
                            points[-1, 0] = x2
                            points[-2, 0] = x2
                            if dtheta[-1] == 45:
                                points[-2, 1] += dy2
                            elif dtheta[-1] == -45:
                                points[-2, 1] -= dy2

            points, normals, ds, theta, dtheta = _compute_segments(points)

            if np.any(np.abs(np.abs(dtheta) - 180) < 1e-6):
                raise ValueError(
                    "smooth() received points which double-back on themselves"
                    "--turns cannot be computed when going forwards then exactly backwards."
                )

            # FIXME add caching
            # Create arcs
            paths = []
            radii = []
            for dt in dtheta:
                P = bend(radius=radius, angle=dt, **kwargs)
                chord = np.linalg.norm(P.points[-1, :] - P.points[0, :])
                r = (chord / 2) / np.sin(np.radians(dt / 2))
                r = np.abs(r)
                radii.append(r)
                paths.append(P)

            d = np.abs(np.array(radii) / np.tan(np.radians(180 - dtheta) / 2))
            encroachment = np.concatenate([[0], d]) + np.concatenate([d, [0]])
            if np.any(encroachment > ds):
                raise ValueError(
                    "smooth(): Not enough distance between points to to fit curves."
                    "Try reducing the radius or spacing the points out farther"
                )
            p1 = points[1:-1, :] - normals[:-1, :] * d[:, np.newaxis]

            # Move arcs into position
            new_points = []
            new_points.append([points[0, :]])
            for n in range(len(dtheta)):
                P = paths[n]
                P.rotate(theta[n] - 0)
                P.move(p1[n])
                new_points.append(P.points)
            new_points.append([points[-1, :]])
            new_points = np.concatenate(new_points)

            start_index = 0
            while 1:
                x1, y1 = new_points[start_index]
                x2, y2 = new_points[start_index + 1]
                dx = abs(x1 - x2)
                dy = abs(y1 - y2)
                if dx < 1e-9 and dy < 1e-9:
                    start_index += 1
                else:
                    break
            end_index = -1

            x1, y1 = new_points[end_index]
            x2, y2 = new_points[end_index - 1]
            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            if dx < 1e-6 and dy < 1e-6:
                P = Path(new_points[start_index:-1, :])
            else:
                P = Path(new_points[start_index:, :])

            return P, points, np.sum(np.abs(dtheta))
        except:
            points = cp
            points, normals, ds, theta, dtheta = _compute_segments(points)
            colinear_elements = np.concatenate(
                [[False], np.abs(dtheta) < 1e-6, [False]]
            )
            if np.any(colinear_elements):
                new_points = points[~colinear_elements, :]
                points, normals, ds, theta, dtheta = _compute_segments(new_points)

            if np.any(np.abs(np.abs(dtheta) - 180) < 1e-6):
                raise ValueError(
                    "smooth() received points which double-back on themselves"
                    "--turns cannot be computed when going forwards then exactly backwards."
                )

            # FIXME add caching
            # Create arcs
            paths = []
            radii = []
            for dt in dtheta:
                P = bend(radius=radius, angle=dt, **kwargs)
                chord = np.linalg.norm(P.points[-1, :] - P.points[0, :])
                r = (chord / 2) / np.sin(np.radians(dt / 2))
                r = np.abs(r)
                radii.append(r)
                paths.append(P)

            d = np.abs(np.array(radii) / np.tan(np.radians(180 - dtheta) / 2))
            encroachment = np.concatenate([[0], d]) + np.concatenate([d, [0]])
            if np.any(encroachment > ds):
                raise ValueError(
                    "smooth(): Not enough distance between points to to fit curves."
                    "Try reducing the radius or spacing the points out farther"
                )
            p1 = points[1:-1, :] - normals[:-1, :] * d[:, np.newaxis]

            # Move arcs into position
            new_points = []
            new_points.append([points[0, :]])
            for n in range(len(dtheta)):
                P = paths[n]
                P.rotate(theta[n] - 0)
                P.move(p1[n])
                new_points.append(P.points)
            new_points.append([points[-1, :]])
            new_points = np.concatenate(new_points)

            start_index = 0
            while 1:
                x1, y1 = new_points[start_index]
                x2, y2 = new_points[start_index + 1]
                dx = abs(x1 - x2)
                dy = abs(y1 - y2)
                if dx < 1e-9 and dy < 1e-9:
                    start_index += 1
                else:
                    break
            end_index = -1

            x1, y1 = new_points[end_index]
            x2, y2 = new_points[end_index - 1]
            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            if dx < 1e-6 and dy < 1e-6:
                P = Path(new_points[start_index:-1, :])
            else:
                P = Path(new_points[start_index:, :])

            return P, points, np.sum(np.abs(dtheta))
    else:
        points = cp
        points, normals, ds, theta, dtheta = _compute_segments(points)
        colinear_elements = np.concatenate([[False], np.abs(dtheta) < 1e-6, [False]])
        if np.any(colinear_elements):
            new_points = points[~colinear_elements, :]
            points, normals, ds, theta, dtheta = _compute_segments(new_points)

        if np.any(np.abs(np.abs(dtheta) - 180) < 1e-6):
            raise ValueError(
                "smooth() received points which double-back on themselves"
                "--turns cannot be computed when going forwards then exactly backwards."
            )

        # FIXME add caching
        # Create arcs
        paths = []
        radii = []
        for dt in dtheta:
            P = bend(radius=radius, angle=dt, **kwargs)
            chord = np.linalg.norm(P.points[-1, :] - P.points[0, :])
            r = (chord / 2) / np.sin(np.radians(dt / 2))
            r = np.abs(r)
            radii.append(r)
            paths.append(P)

        d = np.abs(np.array(radii) / np.tan(np.radians(180 - dtheta) / 2))
        encroachment = np.concatenate([[0], d]) + np.concatenate([d, [0]])
        if np.any(encroachment > ds):
            raise ValueError(
                "smooth(): Not enough distance between points to to fit curves."
                "Try reducing the radius or spacing the points out farther"
            )
        p1 = points[1:-1, :] - normals[:-1, :] * d[:, np.newaxis]

        # Move arcs into position
        new_points = []
        new_points.append([points[0, :]])
        for n in range(len(dtheta)):
            P = paths[n]
            P.rotate(theta[n] - 0)
            P.move(p1[n])
            new_points.append(P.points)
        new_points.append([points[-1, :]])
        new_points = np.concatenate(new_points)

        start_index = 0
        # points_len = len(new_points)
        while 1:
            x1, y1 = new_points[start_index]
            x2, y2 = new_points[start_index + 1]
            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            if dx < 1e-9 and dy < 1e-9:
                start_index += 1
            else:
                break
        end_index = -1

        x1, y1 = new_points[end_index]
        x2, y2 = new_points[end_index - 1]
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        if dx < 1e-6 and dy < 1e-6:
            P = Path(new_points[start_index:-1, :])
        else:
            P = Path(new_points[start_index:, :])

        return P, points, np.sum(np.abs(dtheta))


def alignment(origin_path, port1, port2, radius, bend):
    path = copy.deepcopy(origin_path)
    orient1 = port1.orientation
    orient2 = port2.orientation
    if orient1 == 0 or orient1 == 180:
        path[0][1] = port1.dcenter[1]
        path[1][1] = port1.dcenter[1]
    else:
        path[0][0] = port1.dcenter[0]
        path[1][0] = port1.dcenter[0]
    if orient2 == 0 or orient2 == 180:
        path[-1][1] = port2.dcenter[1]
        path[-2][1] = port2.dcenter[1]
    else:
        path[-1][1] = port2.dcenter[1]
        path[-2][1] = port2.dcenter[1]
    try:
        smooth(
            points=path,
            radius=radius - 1e-3,
            bend=bend,
            # bend=gf.path.arc,
            use_eff=True,
            alignment=False,
        )
        return path
    except:
        return None
    # else:
    #     path[0][0] = port1.dcenter[0]
    #     path[1][0] = port1.dcenter[0]
    #     try:
    #         smooth(points=path,
    #                 radius=radius-1e-3,
    #                 bend=bend,
    #                 # bend=gf.path.arc,
    #                 use_eff=True,
    #                 alignment=False)
    #     except:
    #         return None
    # ori2 = port2.orientation
    # if ori2 == 0 or ori2 == 180:
    #     path[-1][1] = port2.dcenter[1]
    #     path[-2][1] = port2.dcenter[1]
    #     try:
    #         smooth(points=path,
    #                 radius=radius-1e-3,
    #                 bend=bend,
    #                 # bend=gf.path.arc,
    #                 use_eff=True,
    #                 alignment=False)
    #     except:
    #         return None
    # else:
    #     path[-1][0] = port2.dcenter[0]
    #     path[-2][0] = port2.dcenter[0]
    #     try:
    #         smooth(points=path,
    #                 radius=radius-1e-3,
    #                 bend=bend,
    #                 # bend=gf.path.arc,
    #                 use_eff=True,
    #                 alignment=False)
    #     except:
    #         return None
