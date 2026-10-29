# Handoff
## Current state
Basic 2D FK sim in progress (plan: ~/.claude/plans/declarative-prancing-sonnet.md). Done: venv + requirements, `src/lattice.py` (+ tests, 3 passing).
## Recent decisions
- Particles = hexagonal-cell centres on a rectangular triangular-lattice patch, 6 NN harmonic springs.
- Triangular substrate (minima at cell centres), mismatch a/a_s a free parameter; start commensurate.
- Overdamped T=0 dynamics; NO periodic boundaries; outer ring fixed via a `mobile` mask.
- Voronoi tessellation of centres to recover cells.
- Substrate formula: V = (2V0/9)(3 − Σcos(G_i·r)), |G| = 4π/(√3 a_s) (corrects plan).
## Next steps
1. User reviews `src/lattice.py`.
2. Substrate → springs → driving/forces/integrators → tessellation → observables/simulation/io/main (one module at a time, explained with tests, outputs, plots).
## Unverified #WARN: comments
- `src/lattice.py` `triangular_patch`: lattice geometry (row spacing, odd-row shift, zig-zag edges).
- `src/lattice.py` `bonds_nn`: distance-based bonds; undeformed lattice only, tol ≪ 0.73.
## Open questions
- (none)
