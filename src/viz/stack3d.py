#INFO: Interactive 3D stack of the particles' Voronoi cells over time (z = t), like a comb growing upward; plotly, saved as HTML


import numpy as np
import plotly.graph_objects as go

from src.viz.basic_fk import particle_cells

LEFT_COLOR = "#d62728" # particle out of its starting well (red)
STAYED_COLOR = "#9ecae1" # particle still in its starting well (light blue)
PLAIN_COLOR = "#e3c878" # coloring switched off (wax yellow)
EDGE_COLOR = "#3b2f0b" # cell walls (dark brown)


def layer_indices(times, n_layers):
    """
    Indices of the snapshots closest to n_layers evenly spaced times from the first to the last snapshot.

    Parameters
    ----------
    times: Array of shape (S,), the snapshot times (increasing)
    n_layers: Number of layers wanted

    Returns
    ----------
    idx: Array of distinct snapshot indices, increasing (fewer than n_layers if there are fewer snapshots)
    """
    targets = np.linspace(times[0], times[-1], n_layers)
    idx = np.unique([int(np.argmin(np.abs(times - t))) for t in targets])
    return idx


def cell_mesh(cells, z):
    """
    Flat triangle fans filling the polygons cells at height z, in the form plotly's Mesh3d wants.

    Parameters
    ----------
    cells: List of arrays, cells[n] holds the corners (x, y) of the n-th cell in order
    z: Height of the layer

    Returns
    ----------
    x, y, zs: Arrays of the mesh vertices' coordinates
    i, j, k: Arrays of vertex indices, triangle t has corners i[t], j[t], k[t]
    owner: Array, owner[t] = n means triangle t belongs to cells[n]
    """
    x, y, i, j, k, owner = [], [], [], [], [], []
    for n, c in enumerate(cells):
        base = len(x)
        x.extend(c[:, 0])
        y.extend(c[:, 1])
        for m in range(1, len(c) - 1): # fan from corner 0: (0, m, m + 1)
            i.append(base)
            j.append(base + m)
            k.append(base + m + 1)
            owner.append(n)
    zs = np.full(len(x), z)
    return np.array(x), np.array(y), zs, np.array(i), np.array(j), np.array(k), np.array(owner)


def cell_walls(cells, z):
    """
    Outline of every cell at height z as one line (cells separated by None, so plotly does not join them).

    Parameters
    ----------
    cells: List of arrays, cells[n] holds the corners (x, y) of the n-th cell in order
    z: Height of the layer

    Returns
    ----------
    x, y, zs: Lists of coordinates (with None between cells)
    """
    x, y, zs = [], [], []
    for c in cells:
        closed = np.vstack([c, c[:1]]) # back to the first corner
        x.extend(closed[:, 0].tolist() + [None])
        y.extend(closed[:, 1].tolist() + [None])
        zs.extend([z] * len(closed) + [None])
    return x, y, zs


def plot_stack(times, snapshots, left, window, n_layers=8, title=""):
    """
    3D stack of the Voronoi cells of the particles inside the window, one layer per chosen snapshot at height z = t.
    A slider builds the stack layer by layer; a button switches the "out of starting well" coloring on and off.

    Parameters
    ----------
    times: Array of shape (S,), the snapshot times
    snapshots: Array of shape (S, N, 2), snapshots[n] gives the positions at times[n]
    left: Boolean array of shape (S, N), left[n, k] is True if particle k is out of its starting well at times[n]
    window: Tuple (x_min, x_max, y_min, y_max), only particles inside it are drawn
    n_layers: Number of layers (snapshots evenly spaced in time)
    title: Figure title

    Returns
    ----------
    fig: The plotly figure (save with fig.write_html)
    """
    idx = layer_indices(times, n_layers)
    traces = []
    colored, plain = [], [] # facecolor arrays of the mesh traces for the two button states (None for wall traces)

    for n in idx:
        cells, which = particle_cells(snapshots[n], window, pad=0.0) # cells of the particles inside the window at this time
        z = times[n]
        x, y, zs, i, j, k, owner = cell_mesh(cells, z)
        col = np.where(left[n][which][owner], LEFT_COLOR, STAYED_COLOR)
        traces.append(go.Mesh3d(x=x, y=y, z=zs, i=i, j=j, k=k, facecolor=col, flatshading=True, opacity=0.85,
                                hoverinfo="skip", name=f"t = {z:.3g}", showlegend=False))
        colored.append(col)
        plain.append(np.full(len(col), PLAIN_COLOR))
        wx, wy, wz = cell_walls(cells, z)
        traces.append(go.Scatter3d(x=wx, y=wy, z=wz, mode="lines", line=dict(color=EDGE_COLOR, width=2),
                                   hoverinfo="skip", showlegend=False))
        colored.append(None)
        plain.append(None)

    fig = go.Figure(traces)
    n_tr = len(traces)

    #INFO: Slider: step m shows layers 0..m (2 traces per layer: cells and walls); starts with the whole stack visible
    steps = []
    for m, n in enumerate(idx):
        visible = [t < 2 * (m + 1) for t in range(n_tr)]
        steps.append(dict(method="restyle", args=[{"visible": visible}], label=f"{times[n]:.3g}"))
    sliders = [dict(active=len(idx) - 1, currentvalue=dict(prefix="up to t = "), pad=dict(t=30), steps=steps)]

    #INFO: Button: restyle only the mesh traces' facecolor, so it does not change which layers the slider shows
    mesh_ids = list(range(0, n_tr, 2))
    buttons = [
        dict(label="Color: out of well", method="restyle", args=[{"facecolor": [colored[t] for t in mesh_ids]}, mesh_ids]),
        dict(label="Color: off", method="restyle", args=[{"facecolor": [plain[t] for t in mesh_ids]}, mesh_ids]),
    ]
    updatemenus = [dict(type="buttons", direction="right", x=0.0, y=1.08, xanchor="left", buttons=buttons)]

    fig.update_layout(
        title=title + "<br><sup>red = out of starting well, blue = still in it</sup>",
        sliders=sliders, updatemenus=updatemenus,
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="t", aspectmode="manual", aspectratio=dict(x=1, y=1, z=1)),
        margin=dict(l=0, r=0, t=80, b=0),
    )
    return fig
