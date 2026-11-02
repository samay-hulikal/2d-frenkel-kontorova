# Handoff
## Current state
Basic 2D FK sim in progress. 9 tests passing.
- `src/lattice.py` (user-reviewed): `triangular_patch`, `bonds_nn`, `edge_mask`, + new `mobile_mask(nx, ny, fix_edges)`.
- `src/substrate.py` (user-reviewed): `reciprocal_vectors`, `substrate_energy`, `substrate_force`.
- `coordination` is a test-only helper in `tests/test_lattice.py`.
## Recent decisions
- Particles = hexagonal-cell centres on a rectangular triangular-lattice patch, 6 NN harmonic springs — standard 2D FK geometry.
- `pos[k]`, k = j*nx + i; `bonds[l] = [p, q]`; `mask[k]` — one flat index per particle, simpler for forces/masks.
- Bonds built once on the undeformed lattice and never rewired — FK springs don't break.
- Edges: simple on/off `fix_edges` flag (not per-side) — user unsure which, wants to switch easily.
- Triangular substrate, minimum at origin, same orientation as the patch — a = a_s is commensurate.
- V = (2V0/9)(3 − Σcos(2π G_i·r)), G_i·v_j = δ_ij so |G| = 2/(√3 a_s); V0 = peak height (hop barrier = 8V0/9) — user's choice.
- G3 = −(G1 + G2) (equal-length, 120° apart) — needed for sixfold-symmetric barriers.
- Substrate phases/forces written component by component, no matrix products — user prefers readability over speed.
- Overdamped T=0 dynamics; no periodic boundaries.
- Voronoi tessellation of centres to recover cells (also gives geometric neighbour counts for defects).
- Docstring style (Parameters/Returns) recorded in CLAUDE.md Conventions.
## Next steps
1. Springs → driving/forces/integrators → tessellation → observables/simulation/io/main (one module at a time, explained, with tests).
## Unverified #WARN: comments
- (none)
## Open questions
- (none)
