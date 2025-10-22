"""Helper to run repository-level scripts."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def run_script(script_name: str) -> None:
    """Load and run a script by filename."""
    script_path = Path(__file__).resolve().parents[3] / "scripts" / script_name
    spec = spec_from_file_location("llmath_script", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load script: {script_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "main"):
        module.main()
    else:
        raise AttributeError(f"Script has no main(): {script_path}")
