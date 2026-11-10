import matplotlib
matplotlib.use("Agg") # draw without opening windows
import matplotlib.pyplot as plt

from src.viz.transition import plot_transition


def test_one_line_per_V0_and_edge_type():
    rows = [{"V0": V0, "s": s, "a_ratio": s**-0.5, "fix_edges": fe, "frac_left_window": 0.0, "converged": True}
            for V0 in (0.1, 0.5) for s in (0.6, 1.0, 2.0) for fe in (False, True)]
    for x in ("s", "a_ratio"):
        fig = plot_transition(rows, x)
        assert len(fig.axes[0].get_legend().get_texts()) == 4
        plt.close(fig)
