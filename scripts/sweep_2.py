#INFO: Second sweep: same V0 and s values as scripts/sweep.py, but every particle starts in a substrate minimum (spacing a_s);
# results in outputs/sweep_2/. The window is the central quarter of the relaxed patch, so it holds about 30 x 30 particle cells


from pathlib import Path

from scripts.sweep import run_sweep

OUTPUT_DIR = Path("outputs") / "sweep_2"


if __name__ == "__main__":
    run_sweep(OUTPUT_DIR, start_at_minima=True)
