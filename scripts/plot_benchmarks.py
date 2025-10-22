#!/usr/bin/env python3
"""Plot benchmark results into charts."""

import argparse
import json
import sys
from pathlib import Path


def load_results(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def plot_bar(values: dict[str, float | None], title: str, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print("matplotlib is required for plotting:", exc)
        sys.exit(1)

    labels = list(values.keys())
    data = [values[label] if values[label] is not None else 0 for label in labels]

    plt.figure(figsize=(8, 4))
    plt.bar(labels, data, color="#1f77b4")
    plt.title(title)
    plt.ylabel(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument(
        "--input",
        type=str,
        default="benchmarks/results.json",
        help="Path to benchmark results JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/charts",
        help="Output directory for charts",
    )
    args = parser.parse_args()

    results = load_results(Path(args.input))
    modes = results.get("modes", {})

    accuracy = {mode: data["metrics"]["accuracy"] for mode, data in modes.items()}
    token_eff = {
        mode: data["metrics"]["token_efficiency"] for mode, data in modes.items()
    }

    output_dir = Path(args.output_dir)
    plot_bar(accuracy, "Accuracy", output_dir / "accuracy.png")
    plot_bar(token_eff, "Token Efficiency", output_dir / "token_efficiency.png")

    print(f"Charts saved to: {output_dir}")


if __name__ == "__main__":
    main()
