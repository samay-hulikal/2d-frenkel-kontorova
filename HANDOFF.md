# Handoff
## Current state
Basic 2D FK sim in progress. 13 tests passing.
- `src/lattice.py` (user-reviewed): `triangular_patch`, `bonds_nn`, `edge_mask`, `mobile_mask(nx, ny, fix_edges)`.
- `src/substrate.py` (user-reviewed): `reciprocal_vectors`, `substrate_energy`, `substrate_force`.
- `src/springs.py` (committed but NOT yet user-reviewed; WARNs open): `bond_vectors`, `spring_energy` (total), `spring_force`.
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
- Springs: E = (k/2) Σ (|r_q − r_p| − a)², rest length a = bees' natural spacing.
- Relaxation: L-BFGS (scipy) over mobile coordinates — user's choice; overdamped stepping later for driving.
- Initial state: patch at substrate spacing a_s (all particles in minima, springs strained) — user's choice.
- Voronoi tessellation of centers to recover cells (also gives geometric neighbor counts for defects).
- Docstring style (Parameters/Returns) recorded in CLAUDE.md Conventions.
## Next steps
1. User reviews `src/springs.py` and its #WARNs.
2. `src/relax.py`: total energy/force, `relax()` via L-BFGS on mobile particles, stop on max force.
3. `scripts/main.py`: parameters, build → relax → print summary, save plot to outputs/.
4. Later: overdamped dynamics + driving, tessellation, observables.
## Unverified #WARN: comments
- `src/springs.py` `spring_force`: single-bond force −k(|d| − a) d/|d| on q, opposite on p.
- `src/springs.py` `spring_force`: summing over bonds with `np.add.at`.
## Open questions
- (none)
