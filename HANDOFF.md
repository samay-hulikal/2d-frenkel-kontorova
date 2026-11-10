# Handoff
## Current state
Basic 2D FK sim + two parameter sweeps. 30 tests passing.
- `src/lattice.py` (user-reviewed except new `triangles`, `flipped_triangles`; WARN open): `triangular_patch`, `bonds_nn`, `edge_mask`, `mobile_mask(nx, ny, fix_edges)`, `triangles(bonds)`, `flipped_triangles(pos0, pos, tri)`.
- `src/substrate.py` (user-reviewed): `reciprocal_vectors`, `substrate_energy`, `substrate_force`.
- `src/springs.py` (user-reviewed): `bond_vectors`, `spring_energy` (total `V_int`), `spring_force` (`F_int`, built from the force on p, the tail of bond p→q).
- `src/relax.py` (user-reviewed except new `relax_ramped`; WARN open): `total_energy`, `total_force`, `relax` (L-BFGS-B on mobile particles, returns `pos, info`), `relax_ramped` (rest length a_start → a in `n_stages` full relaxes).
- `src/stress.py` (committed, NOT yet user-reviewed; WARNs open): `voronoi_areas` (NaN if unbounded), `isotropic_stress`.
- `src/plotting.py` (committed, NOT yet user-reviewed; WARN open, user: not critical for now): `central_window`, `in_window`, `substrate_cells`, `particle_cells`, `plot_state` (layout approved by user).
- `scripts/main.py`: constants at top (A = √3, S, FIX_EDGES, START_SPACING, N_STAGES, ...); `run(...)` builds → `relax_ramped` from START_SPACING → flipped-triangle count → σ_iso → summary dict (shared with sweeps); `main()` prints and saves `outputs/state_s<S>_V0<V0>_<edges>_start<x>a.png`. Defaults S = 0.75, V0 = 0.1 (placeholder), free edges, start at a.
- `scripts/sweep.py`: `run_sweep(output_dir, start_at_minima)`; V0 = 0.005, 0.255, …, 2.005 × s = 0.5, 0.75, 1, 1.5, 2, 3 (54 runs); free edges; start at a → `outputs/sweep/` (~20 s). Figures `V0_<V0>_s_<s>.png` + `summary.csv` (now with `n_flipped`, `n_window`).
- `scripts/sweep_2.py`: same sweep, start in the minima (spacing a_s) → `outputs/sweep_2/` (~76 s, the ramp needs many iterations when the lattice floats).
- `scripts/sweep_3.py`: V0 = 1e-7, 1e-6, 1e-5, 1e-4 at s = 3, both starts → `outputs/sweep_3/start_relaxed/` and `start_minima/` (~25 s). `run_sweep` now takes `V0_values`, `s_values`; filenames use `:g`. Result: fully floating, σ_iso ≈ 0, E ≈ N·2V0/3. At V0 ≤ 1e-6 the substrate forces are below f_tol = 1e-6, so the start-relaxed runs barely move (0–1 iterations). (Earlier run, V0 = 0.0005–0.01 linear: also fully floating.)
- Latest sweeps: all 108 runs converged, 0 flipped triangles, window 826–1056 particles (mostly 900). Sweep 1 unchanged by the ramp. Sweep 2 at V0 = 0.005 now floats like sweep 1 for every s (E ≈ 12); locks to the substrate (uniform σ_iso = √3 k (a_s − a)/a_s) from V0 ≈ 0.255 for s = 0.5, 0.75, 1.5; 0.755 for s = 2; 1.755 for s = 3.
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
- Ramped relax (20 stages, rest length from the start spacing to a) in all runs: a single relax from a strongly strained start (s = 3, start at a_s, 73% stretch) folded the patch's edges over (370–530 flipped triangles, E = 24 even at V0 = 0). Flipped triangles are counted in every run — user's choice.
- Initial state: springs relaxed (patch at spacing a), particle at origin in a minimum — user's choice for sweep and main; `START_SPACING = A_S` puts all particles in minima instead (sweep 2).
- Window = central 0.25× area of the RELAXED patch's bounding box (≈ 30×30 particle cells) — user's choice; changed from the starting patch, which held 420–3010 particles in sweep 2.
- Figure: left blue substrate hexagons over thicker dark-yellow particle Voronoi, black dots, no potential map; right particle Voronoi filled by σ_iso (positive = expanded), `PRGn` centered at 0, scaled to the window.
- Substrate stiffness at a minimum k_s = 16π²V0/(9a_s²) vs ~3k from springs (rough guide; user to check).
- Voronoi tessellation of centers to recover cells (also gives geometric neighbor counts for defects).
- Docstring style (Parameters/Returns) recorded in CLAUDE.md Conventions.
## Next steps
1. User looks at the sweep figures/summaries in `outputs/sweep/` and `outputs/sweep_2/`.
2. User reviews #WARNs below (plotting #WARN can wait).
3. User picks V0 from the equations / paper.
4. Later: overdamped dynamics + driving, observables.
## Unverified #WARN: comments
- `src/stress.py` `isotropic_stress`: virial stress σ = (1/A) Σ k dl |d| d̂⊗d̂ (springs only), positive = expanded.
- `src/stress.py` `isotropic_stress`: half of each bond to each end; σ_iso = T_k / (2 A_k).
- `src/plotting.py` `substrate_cells`: hexagon corners at distance a_s/√3, angles 30° + 60°n (= maxima).
- `src/lattice.py` `flipped_triangles`: orientation test by the sign of (r_q − r_p) × (r_r − r_p) vs the starting patch.
- `src/relax.py` `relax_ramped`: rest length a_n in equal steps from a_start to a, each stage from the previous result; only the last stage is the real model.
## Open questions
- Boundary conditions: free vs fixed switchable; periodic is a possible later option.
- L-BFGS finds a local minimum; result depends on the start (spacing, registry of the patch on the substrate) and the relax path.
- s = 1 with start at a is trivially E = 0 (patch already in the minima, nothing moves).
- Small rotations (Novaco–McTague) are allowed with free edges but unlikely from an aligned start; could measure the window lattice's angle or start rotated.
