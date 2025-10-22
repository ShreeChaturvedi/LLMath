"""Entry point wrapper for scripts/run_benchmark.py."""

from ._runner import run_script


def main() -> None:
    run_script("run_benchmark.py")
