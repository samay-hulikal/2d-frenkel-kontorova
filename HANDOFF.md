# Handoff
## Current state
Basic 2D FK sim in progress. Done: venv + requirements, `src/lattice.py` (reviewed by user; 3 tests passing).
`src/lattice.py` = `triangular_patch`, `bonds_nn`, `edge_mask`. `coordination` moved to `tests/test_lattice.py` (test-only helper).
## Recent decisions
- Particles = hexagonal-cell centres on a rectangular triangular-lattice patch, 6 NN harmonic springs — standard 2D FK geometry.
- `pos[k]`, k = j*nx + i; `bonds[l] = [p, q]`; `mask[k]` — one flat index per particle, simpler for forces/masks.
- Bonds built once on the undeformed lattice and never rewired — FK springs don't break.
- Triangular substrate (minima at cell centres), mismatch a/a_s free; start commensurate.
- Substrate formula: V = (2V0/9)(3 − Σcos(G_i·r)), |G| = 4π/(√3 a_s).
- Overdamped T=0 dynamics; no periodic boundaries; motion restricted by a `mobile` mask.
- Voronoi tessellation of centres to recover cells (also gives geometric neighbour counts for defects).
- Docstring style (Parameters/Returns) recorded in CLAUDE.md Conventions.
## Next steps
1. Settle edge handling (see Open questions), then add `mobile_mask` to `src/lattice.py`.
2. Substrate → springs → driving/forces/integrators → tessellation → observables/simulation/io/main (one module at a time, explained, with tests).
## Unverified #WARN: comments
- (none)
## Open questions
- Edges fixed or free: simple `fix_edges` on/off flag, or per-side switch (e.g. fix only left/right)?
