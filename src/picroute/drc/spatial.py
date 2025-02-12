"""
Date: 2024-06-05 00:16:09
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-06-05 00:16:09
FilePath: /PICRoute/src/picroute/drc/spatial.py
"""

from rtree import index


class Spatial:
    def __init__(self):
        self._rtree = index.Index(properties=index.Property())

    # get function
    def empty(self):
        pass

    def size(self):
        return len(self._rtree)

    # set function
    def clear(self):
        pass

    def insert(self, id, coord, obj=None):
        self._rtree.insert(id, coord, obj)

    def erase(self):
        pass

    # query
    def query(self, coord):
        return [n.object for n in self._rtree.intersection(coord, objects=True)]

    def nearestSearch(self):
        pass
