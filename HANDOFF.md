# Handoff
## Current state
Reorganized into two models; basic FK now runs as overdamped ODE dynamics (L-BFGS removed, last in commit f094116). 39 tests passing.
- Shared: `src/substrate.py` (user-reviewed), `src/window.py` (`central_window`, `in_window`), `src/dynamics.py` (`overdamped(force, pos0, mobile, ...)`: solve_ivp RK45, stops at max|F| < f_tol or t_max, snapshot every dt_snap).
- `src/basic_fk/`: `lattice` (user-reviewed except `triangles`, `flipped_triangles`), `springs` (reviewed), `energy` (`total_energy`, `total_force`, reviewed, unchanged from relax.py), `stress` (WARNs open), `wells` (`nearest_minimum`, `fraction_left_well`), `simulate` (`kick`, `run` → dict with snapshots, `left` per snapshot, summary).
- `src/viz/`: `substrate` (hexagons), `basic_fk` (`particle_cells`, `plot_state`), `transition` (fraction out of well vs s or a/a_s), `stack3d` (plotly 3D Voronoi stack, z = t, slider builds layers, button toggles coloring).
- `src/vertex_fk/`, `scripts/vertex_fk/`, `tests/vertex_fk/`: empty, start tomorrow.
- `scripts/basic_fk/main.py` (thin): defaults s = 0.75, V0 = 0.1, FIXED edges, start in minima, kick 1e-3 a_s → PNG + `_stack.html` in `outputs/basic_fk/`.
- `scripts/basic_fk/sweep_transition.py`: s = 0.6…3 (13, step 0.2) × V0 (PLACEHOLDER 0.05, 0.25, 0.75) × free/fixed edges → `outputs/basic_fk/transition/` (summary.csv, `transition_vs_s.png`, `transition_vs_a_ratio.png`). Not yet run at full size; free-edge runs take 30–90 s each at 60×60.
- Old outputs moved to `outputs/old_basic_fk/`.
## Findings so far (ODE, 60×60 unless noted)
- Commensurate start is an exact equilibrium (max|F| ~ 1e-14) for any s → the random kick is needed.
- Fixed edges: kick decays (locked) at s = 0.6 and 3, even V0 = 0. Stretched side: commensurate is the global minimum (convexity of stretched springs, every bond line ends on pinned particles). Squeezed side: stable at least to s = 0.6.
- Free edges: floats (≈99% out of wells) at V0 ≤ 0.05 for s = 0.6 and 3; 24×24 smoke test with V0 = 0.75 locked at s = 0.6, 1.8, floated at s = 3.
- solve_ivp rtol must be tiny (1e-10): it multiplies absolute coordinates, and rtol = 1e-6 left runs jittering at max|F| ~ 3e-5.
## Recent decisions
- Folders `basic_fk` / `vertex_fk`; shared code at the top of src/, scripts/, tests/; plotting in `src/viz/`; scripts thin.
- Basic FK: overdamped γ dr/dt = F with adaptive RK45 (no fixed step), γ = 1; no rest-length ramp.
- Transition measure: fraction of central-window particles (window of the final patch) whose nearest substrate minimum differs from the start.
- Transition sweep: one set of runs, plotted against s and a/a_s; both edge types.
- 3D stack: layers evenly spaced in t, z = t, slider shows layers 0..m, color red = out of well / blue = in, toggle off = wax yellow.
- Vertex model (from tomorrow): E = Σ K₁/2 (A − A0)² + K₂/2 (P − P0)² + V_new, V_new = V0 − V_old (minima at the substrate hexagon corners, hop barrier V0/9), vertices start on the minima with a small random kick, fixed edges, overdamped ODE. Guards from the start: softened |e| (zero-length edges) and crossed (bow-tie) face detection.
- Earlier decisions (units a = √3, s = area ratio, substrate form, bonds never rewired) unchanged; see git history of this file.
## Next steps
1. User reviews the new #WARNs below and the ODE code.
2. Pick V0 values; run `sweep_transition` at full size.
3. Vertex model: open questions: pure vertex model (no edge springs)? p0 free or from the bee hexagon (3.722 = zero shear stiffness for an ordered sheet)? Fixed connectivity (no T1)? Exact V_new?
## Unverified #WARN: comments (new this session)
- `src/dynamics.py` `overdamped`: equation of motion γ dr/dt = F.
- `src/dynamics.py` `overdamped`: stopping event max|F| − f_tol; tolerance atol + rtol|x| with rtol tiny.
- `src/basic_fk/wells.py` `nearest_minimum`: nearest minimum is one of the 4 corners of the lattice rhombus holding the point.
- `src/basic_fk/simulate.py` `run`: starting kick; wells and displacements measured from the unkicked lattice.
- Older #WARNs remain in untouched code: `stress.py` (2), `viz/substrate.py`, `lattice.py` `flipped_triangles`.
