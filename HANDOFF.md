# Handoff
## Current state
Basic 2D FK sim + first parameter sweep. 26 tests passing.
- `src/lattice.py` (user-reviewed): `triangular_patch`, `bonds_nn`, `edge_mask`, `mobile_mask(nx, ny, fix_edges)`.
- `src/substrate.py` (user-reviewed): `reciprocal_vectors`, `substrate_energy`, `substrate_force`.
- `src/springs.py` (user-reviewed): `bond_vectors`, `spring_energy` (total `V_int`), `spring_force` (`F_int`, built from the force on p, the tail of bond p→q).
- `src/relax.py` (user-reviewed): `total_energy`, `total_force`, `relax` (L-BFGS-B on mobile particles, returns `pos, info`).
- `src/stress.py` (committed, NOT yet user-reviewed; WARNs open): `voronoi_areas` (NaN if unbounded), `isotropic_stress`.
- `src/plotting.py` (committed, NOT yet user-reviewed; WARN open, user: not critical for now): `central_window`, `in_window`, `substrate_cells`, `particle_cells`, `plot_state` (layout approved by user).
- `scripts/main.py`: constants at top (A = √3, S, FIX_EDGES, START_SPACING, ...); `run(...)` builds → relaxes → σ_iso → summary dict (shared with sweep); `main()` prints and saves `outputs/state_s<S>_V0<V0>_<edges>_start<x>a.png`. Defaults S = 0.75, V0 = 0.1 (placeholder), free edges, start at a.
- `scripts/sweep.py`: `run_sweep(output_dir, start_at_minima)`; V0 = 0.05, 0.30, …, 2.05 × s = 0.5, 0.75, 1, 1.5, 2, 3 (54 runs, ~17 s); free edges; as a script starts at a → `outputs/sweep/` (`V0_<V0>_s_<s>.png` + `summary.csv`).
- `scripts/sweep_2.py`: same sweep, start in the minima (spacing a_s) → `outputs/sweep_2/`. Window = central quarter of the starting patch, so always 30×30 particle cells (user's choice); its physical size scales with a_s.
- Both sweeps: all 54 runs converged. Sweep 2 locks to the substrate (uniform σ_iso = √3 k (a_s − a)/a_s in window) once V0 is large enough: s = 0.5, 0.75, 1.5 from V0 ≈ 0.3; s = 2 from 0.8; s = 3 from 1.8. Which start gives the lower final energy depends on (V0, s): start at a_s wins for s = 0.75, 1.5 (V0 ≥ 0.55) and for most s at V0 ≥ 1.8; start at a wins for s = 3 and most of s = 0.5, 2.
- `coordination` is a test-only helper in `tests/test_lattice.py`.
## Recent decisions
- Particles = hexagonal-cell centers on a rectangular triangular-lattice patch, 6 NN harmonic springs — standard 2D FK geometry.
- `pos[k]`, k = j*nx + i; `bonds[l] = [p, q]`; `mask[k]` — one flat index per particle, simpler for forces/masks.
- Bonds built once on the undeformed lattice and never rewired — FK springs don't break.
- Edges: simple on/off `fix_edges` flag (not per-side) — user unsure which, wants to switch easily.
- Triangular substrate, minimum at origin, same orientation as the patch — a = a_s is commensurate.
- V = (2V0/9)(3 − Σcos(2π G_i·r)), G_i·v_j = δ_ij so |G| = 2/(√3 a_s); V0 = peak height (hop barrier = 8V0/9) — user's choice.
- G3 = −(G1 + G2) (equal-length, 120° apart) — needed for sixfold-symmetric barriers.
- Code written component by component, no matrix products — user prefers readability over speed.
- Springs: V_int = (k/2) Σ (|r_q − r_p| − a)², rest length a = bees' natural spacing.
- Units (to match a paper): bee cell side length 1 → a = √3; k = 1. s = substrate cell area / bee cell area, so a_s = a√s = √(3s); s = 1 commensurate.
- Relaxation: L-BFGS (scipy) over mobile coordinates — user's choice; overdamped stepping later for driving.
- relax default f_tol = 1e-6: 1e-8 hits the float precision floor on 60×60 (line search "ABNORMAL") — user's choice.
- Initial state: springs relaxed (patch at spacing a), particle at origin in a minimum — user's choice for sweep and main; `START_SPACING = A_S` puts all particles in minima instead.
- Window = 30×30 bee cells: 60×60 patch, central 0.25× area of the starting patch, fixed in space.
- Figure: left blue substrate hexagons over thicker dark-yellow particle Voronoi, black dots, no potential map; right particle Voronoi filled by σ_iso (positive = expanded), `PRGn` centered at 0, scaled to the window.
- Substrate stiffness at a minimum k_s = 16π²V0/(9a_s²) vs ~3k from springs (rough guide; user to check).
- Voronoi tessellation of centers to recover cells (also gives geometric neighbor counts for defects).
- Docstring style (Parameters/Returns) recorded in CLAUDE.md Conventions.
## Next steps
1. User looks at the sweep figures/summaries in `outputs/sweep/` and `outputs/sweep_2/`.
2. User reviews `src/stress.py` #WARNs (plotting #WARN can wait).
3. User picks V0 from the equations / paper.
4. Later: overdamped dynamics + driving, observables.
## Unverified #WARN: comments
- `src/stress.py` `isotropic_stress`: virial stress σ = (1/A) Σ k dl |d| d̂⊗d̂ (springs only), positive = expanded.
- `src/stress.py` `isotropic_stress`: half of each bond to each end; σ_iso = T_k / (2 A_k).
- `src/plotting.py` `substrate_cells`: hexagon corners at distance a_s/√3, angles 30° + 60°n (= maxima).
## Open questions
- Boundary conditions: free vs fixed switchable; periodic is a possible later option.
- L-BFGS finds a local minimum; result depends on the start (spacing, registry of the patch on the substrate).
- s = 1 with start at a is trivially E = 0 (patch already in the minima, nothing moves).
