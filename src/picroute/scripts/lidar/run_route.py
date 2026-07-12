"""
Date: 1969-12-31 17:00:00
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-09-28 05:02:20
FilePath: /PICRoute/src/picroute/scripts/APR/run_route.py
"""

import os
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

from picroute.utils.general import ensure_dir, logger

method = "LiDAR"
exp_name = "main_results"
package_root = Path(__file__).resolve().parents[2]
root = package_root / "log" / method / exp_name
benchmark_root = package_root / "benchmarks"
result_root = package_root / "result" / method / exp_name

comp_config = package_root / "config" / f"comp_{method}.yml"
noc_config = package_root / "config" / f"noc_{method}.yml"


def task_launcher(args):
    (
        benchmark,
        config,
        id,
    ) = args
    env = os.environ.copy()
    benchmark = Path(benchmark)
    config = Path(config)
    benchmark_name = benchmark.stem
    method_name = config.stem
    result_path = result_root / f"{benchmark_name}_{method_name}_id-{id}.gds"
    pres = [sys.executable, "-m", "picroute.main.picroute"]

    with open(
        root / f"{benchmark_name}_{method_name}_id-{id}.log",
        "w",
    ) as wfid:
        exp = [
            f"--benchmark={benchmark}",
            f"--config={config}",
            f"--run.output_layout_gds_path={result_path}",
        ]
        logger.info(f"running command {' '.join(pres + exp)}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


def main() -> None:
    ensure_dir(root)
    ensure_dir(result_root)

    tasks = [
        # [f"{benchmark_root}/clements_8x8/clements_8x8.yml", comp_config, 2],
        # [f"{benchmark_root}/clements_16x16/clements_16x16.yml", comp_config, 2],
        [benchmark_root / "multiportmmi_8x8" / "multiportmmi_8x8.yml", comp_config, 2],
        # [f"{benchmark_root}/multiportmmi_16x16/multiportmmi_16x16.yml", comp_config, 2],
        # [f"{benchmark_root}/multiportmmi_32x32/multiportmmi_32x32.yml", comp_config, 2],
        # [f"{benchmark_root}/router/router8x8_north.yml", noc_config, 1],
        # [f"{benchmark_root}/router/router8x8_oneside.yml", noc_config, 1],
        # [f"{benchmark_root}/router/router8x8_corner.yml", noc_config, 1],
        # [f"{benchmark_root}/router/router8x8_pairwise.yml", noc_config, 1],
    ]

    with Pool(1) as p:
        p.map(task_launcher, tasks)


if __name__ == "__main__":
    main()
