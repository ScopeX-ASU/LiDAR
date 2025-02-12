"""
Date: 2024-06-04 22:42:11
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-06-05 00:18:16
FilePath: /PICRoute/src/picroute/routing/drmanager.py
"""

# from pyutils.general import logger
from picroute.database.schematic import CustomSchematic
from picroute.drc.drcmanager import DrcManager
from picroute.utils.general import logger

from .drgridroute import DrGridRoute


class DrManager(object):
    def __init__(self, Cirdb: CustomSchematic, DrcMgr: DrcManager, config):
        self._cirdb = Cirdb
        self._drcmgr = DrcMgr
        self._config = config
        self.build_router()

    def build_router(self):
        if self._config.dr.router == "GridRoute":
            self._router = DrGridRoute(self._cirdb, self._drcmgr, self._config)
        else:
            logger.error(f"Router {self._config.dr.router} not supported")
            raise NotImplementedError

    def solve(self):
        logger.info("Start Detailed Routing")
        return self._router.solve()
