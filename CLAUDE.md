# Project: 2D Frenkel-Kontorova model and its modifications for a hexagonal lattice
## Purpose
Simulating a 2D Frenkel-Kontorova model and its variations for a hexagonal lattice.
## Layout
src/ functions/classes · scripts/ main code · tests/ tests · sample-data/ inputs · outputs/ results
- Two models: `basic_fk/` (triangular spring network) and `vertex_fk/` (vertex model) under src/, scripts/, tests/; shared code at the top of each. Plotting in `src/viz/`. Keep scripts thin.
## Commands
- Run tests: .venv/bin/python -m pytest
- Run basic FK: .venv/bin/python -m scripts.basic_fk.main
## Conventions
- Docstrings: one-line summary, then `Parameters` and `Returns` sections (dashed underline), one `name: description` line per item (see `src/basic_fk/lattice.py`).
## Working rules
- Work on main; small commits with clear messages.
- Explain the plan before editing; verify by running tests.
- Don't open files in sample-data/ or outputs/ (requires my approval); write code that prints short summaries instead.
- Always ask when something is ambiguous.
- Current state lives in HANDOFF.md.
- Keep CLAUDE.md and HANDOFF.md succinct.
- Keep requirements.txt up to date with required packages.
- Run code inside a virtual environment.
- Use #WARN: comments around pieces that are subtle, involve the model's math, describes the optimization process, or anything that could use my attention as a check.
- A #WARN: still in the code = not yet verified by me. Notify me of any remaining ones (e.g. at pickup) and keep the list in HANDOFF.md current.
