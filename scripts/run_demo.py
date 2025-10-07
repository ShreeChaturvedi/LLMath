#!/usr/bin/env python3
"""Launch Gradio demo interface.

Usage:
    python scripts/run_demo.py [--share]
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="Launch Gradio demo"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to trained LoRA adapters (optional)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on",
    )
    args = parser.parse_args()

    from llmath.config import load_config

    # Load config
    config = load_config(args.config)
    if args.adapter_path:
        config.model.adapter_path = Path(args.adapter_path)

    print("=" * 60)
    print("LLMath Gradio Demo")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Adapter: {config.model.adapter_path or 'None (base model)'}")
    print(f"Share: {args.share}")
    print(f"Port: {args.port}")
    print()

    # Import and launch the app
    from llmath.api.gradio_app import create_demo

    demo = create_demo(config)
    demo.launch(
        share=args.share,
        server_port=args.port,
    )


if __name__ == "__main__":
    main()
