# Handoff
## Current state
Basic 2D FK sim in progress. 26 tests passing.
- `src/lattice.py` (user-reviewed): `triangular_patch`, `bonds_nn`, `edge_mask`, `mobile_mask(nx, ny, fix_edges)`.
- `src/substrate.py` (user-reviewed): `reciprocal_vectors`, `substrate_energy`, `substrate_force`.
- `src/springs.py` (user-reviewed): `bond_vectors`, `spring_energy` (total `V_int`), `spring_force` (`F_int`, built from the force on p, the tail of bond p→q).
- `src/relax.py` (user-reviewed): `total_energy`, `total_force`, `relax` (L-BFGS-B on mobile particles, returns `pos, info`).
- `src/stress.py` (committed, NOT yet user-reviewed; WARNs open): `voronoi_areas` (NaN if unbounded), `isotropic_stress`.
- `src/plotting.py` (committed, NOT yet user-reviewed; WARN open, user: not critical for now): `central_window`, `in_window`, `substrate_cells`, `particle_cells`, `plot_state` (layout approved by user).
- `scripts/main.py`: constants at top (incl. `FIX_EDGES`, `START_SPACING`), build → relax → σ_iso → printed summary → `outputs/state_a<A>_V0<V0>_<edges>_start<s>.png`. V0 = 0.1 is a placeholder.
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
- Relaxation: L-BFGS (scipy) over mobile coordinates — user's choice; overdamped stepping later for driving.
- relax default f_tol = 1e-6: 1e-8 hits the float precision floor on 60×60 (E ≈ 52, line search "ABNORMAL") — user's choice.
- Initial state: patch at substrate spacing a_s (all particles in minima, springs strained) — user's choice.
- Voronoi tessellation of centers to recover cells (also gives geometric neighbor counts for defects).
- Docstring style (Parameters/Returns) recorded in CLAUDE.md Conventions.
## Next steps
1. User reviews `src/stress.py` #WARNs (plotting #WARN can wait).
2. User picks V0 (and start spacing) from the equations.
3. Later: overdamped dynamics + driving, observables.
## Unverified #WARN: comments
- `src/stress.py` `isotropic_stress`: virial stress σ = (1/A) Σ k dl |d| d̂⊗d̂ (springs only), positive = expanded.
- `src/stress.py` `isotropic_stress`: half of each bond to each end; σ_iso = T_k / (2 A_k).
- `src/plotting.py` `substrate_cells`: hexagon corners at distance a_s/√3, angles 30° + 60°n (= maxima).
- Figure: central window fixed in space (0.25× area; 60×60 patch → 30×30 window). Left: blue substrate hexagons, thicker dark-yellow particle Voronoi underneath, black dots; no potential map. Right: particle Voronoi filled by σ_iso (positive = expanded), `PRGn` centered at 0, scaled to the window.
- Units: k = a_s = 1 fix energy/length units; physics set by a/a_s and V0/(k a_s²). Substrate stiffness at a minimum k_s = 16π²V0/(9a_s²) vs ~3k from springs; competition near V0/(k a_s²) ~ 0.1–0.2 (rough, user to check).
## Open questions
- Start at spacing a_s (current choice) keeps the interior locked; starting at a with weak V0 gives discommensuration patterns — user to decide.
- Boundary conditions: user unsure; keep fixed (pins mean spacing at a_s, so the commensurate start doesn't move) vs free switchable. Periodic is a possible later option.
- Units: sim is not dimensionless (user note in relax.py); k and V0 must be chosen with care.
- L-BFGS finds a local minimum: starting from the commensurate state, misfit dislocations don't appear on their own.
