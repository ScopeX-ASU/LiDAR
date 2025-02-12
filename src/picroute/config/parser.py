import argparse


class Parser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "--benchmark",
            default="/home/hzhou144/projects/PICRoute/src/picroute/benchmarks/clements_8x8/clements_8x8.yml",
            help="path to benchmark",
        )
        self.parser.add_argument(
            "--config",
            default="/home/hzhou144/projects/PICRoute/src/picroute/config/comp_LiDAR.yml",
            metavar="File",
            help="the router config",
        )
