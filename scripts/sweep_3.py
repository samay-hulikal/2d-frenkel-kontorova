#INFO: Third sweep: very weak substrates only, V0 = 1e-7, 1e-6, 1e-5, 1e-4 at s = 3, from both starting states;
# results in outputs/sweep_3/start_relaxed/ (spacing a) and outputs/sweep_3/start_minima/ (spacing a_s)


from pathlib import Path

import numpy as np

from scripts.sweep import run_sweep

V0_VALUES = 10.0 ** np.arange(-7, -3) # 1e-7, 1e-6, 1e-5, 1e-4
S_VALUES = [3]

OUTPUT_DIR = Path("outputs") / "sweep_3"


if __name__ == "__main__":
    run_sweep(OUTPUT_DIR / "start_relaxed", False, V0_VALUES, S_VALUES)
    run_sweep(OUTPUT_DIR / "start_minima", True, V0_VALUES, S_VALUES)
