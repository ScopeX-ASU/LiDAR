import copy

import numpy as np
from gdsfactory.path import Path, smooth as smooth_path
from gdsfactory.typings import Coordinates


def _compute_segments(points):
    points = np.asarray(points, dtype=float)
    normals = np.diff(points, axis=0)
    normals = (normals.T / np.linalg.norm(normals, axis=1)).T
    dx = np.diff(points[:, 0])
    dy = np.diff(points[:, 1])
    ds = np.sqrt(dx**2 + dy**2)
    theta = np.degrees(np.arctan2(dy, dx))
    dtheta = np.diff(theta)
    dtheta = dtheta - 360 * np.floor((dtheta + 180) / 360)
    return points, normals, ds, theta, dtheta


def prepare_waypoints(points: Coordinates | Path) -> tuple[np.ndarray, float]:
    """Return sparse routing waypoints and their accumulated turning angle.

    Consecutive collinear waypoints are removed using the same tolerance as
    :func:`gdsfactory.path.smooth`. The returned points are routing state, not the
    densely sampled centerline stored in ``Path.points``.
    """
    if isinstance(points, Path):
        points = points.points

    points, _, _, _, dtheta = _compute_segments(points)
    collinear = np.concatenate([[False], np.abs(dtheta) < 1e-6, [False]])
    if np.any(collinear):
        points, _, _, _, dtheta = _compute_segments(points[~collinear, :])

    return points, float(np.sum(np.abs(dtheta)))


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
        path[-1][0] = port2.dcenter[0]
        path[-2][0] = port2.dcenter[0]
    try:
        smooth_path(
            points=path,
            radius=radius - 1e-3,
            bend=bend,
            use_eff=True,
        )
        return path
    except Exception:
        return None
