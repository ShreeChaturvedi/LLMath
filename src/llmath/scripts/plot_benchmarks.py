"""Entry point wrapper for scripts/plot_benchmarks.py."""

from ._runner import run_script


def main() -> None:
    run_script("plot_benchmarks.py")
