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

# from pyutils.general import ensure_dir, logger
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.general import ensure_dir, logger

method = "LiDAR"
exp_name = "main_results"
root = f"log/{method}/{exp_name}"
script = "main/picroute.py"
benchmark_root = "./benchmarks"
result_root = f"result/{method}/{exp_name}"

comp_config = f"./config/comp_{method}.yml"
noc_config = f"./config/noc_{method}.yml"


def task_launcher(args):
    (
        benchmark,
        config,
        id,
    ) = args
    env = os.environ.copy()
    benchmark_name = os.path.basename(benchmark).split(".")[0]
    method_name = os.path.basename(config).split(".")[0]
    result_path = os.path.join(
        result_root, f"{benchmark_name}_{method_name}_id-{id}.gds"
    )
    pres = ["python3", script]

    with open(
        os.path.join(
            root,
            f"{benchmark_name}_{method_name}_id-{id}.log",
        ),
        "w",
    ) as wfid:
        exp = [
            f"--benchmark={benchmark}",
            f"--config={config}",
            f"--run.output_layout_gds_path={result_path}",
        ]
        logger.info(f"running command {' '.join(pres + exp)}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == "__main__":
    ensure_dir(root)

    tasks = [
        # [f"{benchmark_root}/clements_8x8/clements_8x8.yml", comp_config, 2],
        # [f"{benchmark_root}/clements_16x16/clements_16x16.yml", comp_config, 2],
        [f"{benchmark_root}/multiportmmi_8x8/multiportmmi_8x8.yml", comp_config, 2],
        # [f"{benchmark_root}/multiportmmi_16x16/multiportmmi_16x16.yml", comp_config, 2],
        # [f"{benchmark_root}/multiportmmi_32x32/multiportmmi_32x32.yml", comp_config, 2],
        # [f"{benchmark_root}/router/router8x8_north.yml", noc_config, 1],
        # [f"{benchmark_root}/router/router8x8_oneside.yml", noc_config, 1],
        # [f"{benchmark_root}/router/router8x8_corner.yml", noc_config, 1],
        # [f"{benchmark_root}/router/router8x8_pairwise.yml", noc_config, 1],
    ]

    with Pool(1) as p:
        p.map(task_launcher, tasks)
