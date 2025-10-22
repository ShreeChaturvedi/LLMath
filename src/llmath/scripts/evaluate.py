"""Entry point wrapper for scripts/evaluate.py."""

from ._runner import run_script


def main() -> None:
    run_script("evaluate.py")
