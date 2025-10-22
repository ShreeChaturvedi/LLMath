"""Entry point wrapper for scripts/build_index.py."""

from ._runner import run_script


def main() -> None:
    run_script("build_index.py")
