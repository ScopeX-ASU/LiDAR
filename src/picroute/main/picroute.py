"""
Date: 2024-06-04 22:52:57
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-06-04 23:11:00
FilePath: /PICRoute/src/picroute/main/picroute.py
"""

"""
@file flow.py
@author Hongjian ZHOu 
@date 04/21/2024
@brief the PICRoute flow
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")


sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from picroute.config.config import configs
from picroute.config.parser import Parser
from picroute.database.schematic import CustomSchematic
from picroute.drc.drcmanager import DrcManager
from picroute.routing.drmanager import DrManager
from picroute.utils.general import TimerCtx, logger, set_torch_deterministic
from picroute.version import __version__


class PICRoute(object):
    def __init__(self, path, config):
        self.path = path
        self.config = config
        self.print_welcome()
        set_torch_deterministic(config.run.random_state)
        self.initialize()

    def initialize(self):
        """
        brief initialize the routing flow
        """
        self.cirdb = CustomSchematic(self.path, pdk=None)

        # load yaml netlist and visualize the initial layout in .gds / klayout
        with TimerCtx() as t:
            self.cirdb.load_gp()
        self.cirdb.print_benchmark()
        logger.info(f"reading benchmark takes {t.interval:.2f} seconds")

        # Used for DRC check
        with TimerCtx() as t:
            self.drc = DrcManager(self.cirdb, self.config)
        logger.info(f"initialize bitmap takes {t.interval:.2f} seconds")

        self.dr = DrManager(self.cirdb, self.drc, self.config)

    def print_welcome(self):
        """
        brief print welcome message
        """
        content = f"""\
=========================================================================================
                                   PICRoute v{__version__}
                                    Hongjian Zhou
                        Jiaqi Gu (https://scopex-asu.github.io)"""
        print(content)
        print(
            "\n================================== Routing Parameters ===================================\n"
        )
        print(f"Parameters:\n{self.config}", flush=True)

    def run(self):
        """
        The main function to run the routing flow
        """

        # Detailed routing
        with TimerCtx() as t:
            self.dr.solve()
        logger.info(f"detailed routing takes {t.interval:.2f} seconds")

        # visualize the routed layout
        self.cirdb.layout.write_gds(self.config.run.output_layout_gds_path)

        # self.cirdb.layout = self.cirdb.layout.flatten()
        # gdspath = self.cirdb.layout.write_gds(
        #     flatten_offgrid_references=True, logging=False
        # )
        # layout = gf.import_gds(gdspath)
        # layout.write_gds(self.config.run.output_layout_gds_path)
        # layout.show()
        # self.cirdb.rwguide_layout.write_gds(self.config.run.output_routing_gds_path)


def profile(router):
    import cProfile
    import pstats

    with cProfile.Profile() as profile:
        router.run()
    profile_result = pstats.Stats(profile)
    # profile_result.sort_stats(pstats.SortKey.TIME)
    profile_result.sort_stats(pstats.SortKey.CUMULATIVE)
    profile_result.print_stats(20)


if __name__ == "__main__":
    argp = Parser()
    args, opts = argp.parser.parse_known_args()
    configs.load(args.config, recursive=False)
    configs.update(opts)

    router = PICRoute(args.benchmark, configs)
    router.run()

    # profile(router)
