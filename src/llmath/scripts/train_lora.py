"""Entry point wrapper for scripts/train_lora.py."""

from ._runner import run_script


def main() -> None:
    run_script("train_lora.py")
