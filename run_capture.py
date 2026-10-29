"""Run from the source tree without installing the package."""
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from fp_capture_sim.cli import main
    main()
