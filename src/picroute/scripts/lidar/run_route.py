"""Batch entry point for routing one or more benchmark netlists."""

from __future__ import annotations

import argparse
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

from picroute.utils.general import ensure_directory, logger

package_root = Path(__file__).resolve().parents[2]
default_config = package_root / "config" / "comp_LiDAR.yml"
default_benchmark = (
    package_root / "benchmarks" / "multiportmmi_8x8" / "multiportmmi_8x8.yml"
)

type BatchTask = tuple[Path, Path, Path, Path, int]


def task_launcher(task: BatchTask) -> Path:
    benchmark, config, output_directory, log_directory, task_id = task
    benchmark_name = benchmark.stem
    run_name = f"{benchmark_name}_{config.stem}_id-{task_id}"
    result_path = output_directory / f"{run_name}.gds"
    log_path = log_directory / f"{run_name}.log"
    command = [
        sys.executable,
        "-m",
        "picroute.main.picroute",
        f"--benchmark={benchmark}",
        f"--config={config}",
        f"--run.output_layout_gds_path={result_path}",
    ]
    logger.info("running command %s", " ".join(map(str, command)))
    with log_path.open("w", encoding="utf-8") as logfile:
        subprocess.run(
            command,
            check=True,
            stdout=logfile,
            stderr=subprocess.STDOUT,
        )
    return result_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmarks", nargs="*", type=Path, metavar="BENCHMARK")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--output-dir", type=Path, default=Path("result/LiDAR"))
    parser.add_argument("--log-dir", type=Path, default=Path("log/LiDAR"))
    parser.add_argument("--jobs", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.jobs < 1:
        raise ValueError("--jobs must be at least 1")
    benchmarks = args.benchmarks or [default_benchmark]
    output_directory = ensure_directory(args.output_dir)
    log_directory = ensure_directory(args.log_dir)
    tasks = [
        (benchmark, args.config, output_directory, log_directory, task_id)
        for task_id, benchmark in enumerate(benchmarks, start=1)
    ]

    if args.jobs == 1:
        for task in tasks:
            task_launcher(task)
    else:
        with Pool(args.jobs) as pool:
            pool.map(task_launcher, tasks)


if __name__ == "__main__":
    main()
