"""
Date: 2024-06-05 00:17:45
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-06-05 00:17:46
FilePath: /PICRoute/src/picroute/queue/pqueue.py
"""

from heapq import heapify, heappop, heappush

__all__ = ["AdvancedPriorityQueue"]


class AdvancedPriorityQueue(object):
    def __init__(self):
        self._heap = []
        self._table = set()
        self.id_of = lambda x: id(x)

    def put(self, x):
        heappush(self._heap, x)
        self._table.add(self.id_of(x))

    def get(self):
        item = heappop(self._heap)
        self._table.remove(self.id_of(item))
        return item

    def update(self):
        ## update the PriorityQueue to guarantee a correct internal ordering
        ## if you updated a node that is already in the queue, please call this function after you made the modification.
        heapify(self._heap)

    def len(self):
        return len(self._heap)

    def empty(self):
        return len(self._heap) == 0

    def exist(self, x):
        ## check whether a node is in the queue
        return self.id_of(x) in self._table
