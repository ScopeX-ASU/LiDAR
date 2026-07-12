import argparse
from pathlib import Path


DEFAULT_CONFIG = Path(__file__).with_name("comp_LiDAR.yml")


class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "--benchmark",
            required=True,
            type=Path,
            help="path to benchmark",
        )
        self.parser.add_argument(
            "--config",
            default=DEFAULT_CONFIG,
            type=Path,
            metavar="File",
            help="the router config",
        )
