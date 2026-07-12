from pathlib import Path

from picroute.scripts.lidar import run_route


def test_batch_launcher_checks_subprocess_and_writes_to_requested_paths(
    tmp_path: Path, monkeypatch
) -> None:
    benchmark = tmp_path / "example.yml"
    config = tmp_path / "config.yml"
    output_directory = tmp_path / "results"
    log_directory = tmp_path / "logs"
    output_directory.mkdir()
    log_directory.mkdir()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr(run_route.subprocess, "run", fake_run)

    result = run_route.task_launcher(
        (benchmark, config, output_directory, log_directory, 3)
    )

    assert result == output_directory / "example_config_id-3.gds"
    command, kwargs = calls[0]
    assert f"--run.output_layout_gds_path={result}" in command
    assert kwargs["check"] is True
    assert (log_directory / "example_config_id-3.log").is_file()


def test_batch_parser_defaults_to_one_local_worker() -> None:
    args = run_route._parser().parse_args([])

    assert args.jobs == 1
    assert args.output_dir == Path("result/LiDAR")
