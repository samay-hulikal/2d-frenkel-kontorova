#INFO: Plot of the locked -> floating transition: fraction of window particles out of their starting well against s or a / a_s


import numpy as np
import matplotlib.pyplot as plt


def plot_transition(rows, x="s", title=None):
    """
    Fraction of central-window particles out of their starting well, one line per (V0, edges) pair.

    Parameters
    ----------
    rows: List of dictionaries, each with keys "V0", "s", "a_ratio" (= a / a_s), "fix_edges", "frac_left_window", "converged"
    x: Which quantity goes on the horizontal axis, "s" or "a_ratio"
    title: Optional figure title

    Returns
    ----------
    fig: The matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    V0_values = sorted({r["V0"] for r in rows})
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(V0_values)))

    for V0, color in zip(V0_values, colors):
        for fix_edges, style in ((False, "-"), (True, "--")):
            sel = sorted((r for r in rows if r["V0"] == V0 and r["fix_edges"] == fix_edges), key=lambda r: r[x])
            if not sel:
                continue
            xs = [r[x] for r in sel]
            ys = [r["frac_left_window"] for r in sel]
            ax.plot(xs, ys, style, color=color, marker="o", ms=4, label=f"V0 = {V0:g}, edges {'fixed' if fix_edges else 'free'}")
            bad = [(xv, yv) for xv, yv, r in zip(xs, ys, sel) if not r["converged"]]
            if bad:
                ax.plot(*zip(*bad), "x", color="red", ms=8) # runs that hit t_max before reaching f_tol

    ax.axvline(1.0, color="gray", lw=0.8) # commensurate: s = 1, a = a_s
    ax.set_xlabel("s = substrate cell area / bee cell area" if x == "s" else "a / a_s")
    ax.set_ylabel("fraction out of starting well (central window)")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(title if title is not None else "Locked (0) vs floating (≈1); red × = not converged by t_max")
    return fig
