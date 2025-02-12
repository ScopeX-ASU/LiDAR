"""
Date: 1969-12-31 17:00:00
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-02-05 00:37:11
FilePath: /PICRoute/src/picroute/routing/utils.py
"""

import numpy as np

__all__ = ["manhattanDH", "EulerDist", "customH"]


def customH(pos1, pos2, bending_loss: float, resolution: float):
    dx, dy = abs(pos1[0] - pos2[0]) * resolution, abs(pos1[1] - pos2[1]) * resolution
    if dx != 0 and dy != 0:
        if dx > dy:
            return dx - dy + dy * 2**0.5 + bending_loss / 2
        else:
            return dy - dx + dx * 2**0.5 + bending_loss / 2
    else:
        return max(dx, dy)


def manhattanDH(node1, node2):
    return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])


def EulerDist(pos1, pos2):
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5


def vector_intersection(
    center1, ori1, center2, ori2, max_distance=100000, raise_error=True
) -> bool:
    """
    Gets the intersection point between two vectors, specified by (point, angle) pairs, (p0, a0) and (p1, a1).

    Args:
        p0: x,y location of vector 0.
        a0: angle of vector 0 [degrees].
        p1: x,y location of vector 1.
        a1: angle of vector 1 [degrees].
        max_distance: maximum search distance for an intersection [um].
        raise_error: if True, raises an error if no intersection is found. Otherwise, returns None in that case.

    Returns:
        The (x,y) point of intersection, if one is found. Otherwise None.
    """
    import shapely.geometry as sg

    a0_rad = np.deg2rad(ori1)
    a1_rad = np.deg2rad(ori2)
    dx0 = max_distance * np.cos(a0_rad)
    dy0 = max_distance * np.sin(a0_rad)
    p0_far = np.asarray(center1) + [dx0, dy0]
    l0 = sg.LineString([center1, p0_far])

    dx1 = max_distance * np.cos(a1_rad)
    dy1 = max_distance * np.sin(a1_rad)
    p1_far = np.asarray(center2) + [dx1, dy1]
    l1 = sg.LineString([center2, p1_far])

    intersect = l0.intersection(l1)
    if isinstance(intersect, sg.Point):
        return True

    return False
