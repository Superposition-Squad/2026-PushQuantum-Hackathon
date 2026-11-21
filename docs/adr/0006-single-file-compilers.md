# ADR-0006: Every compiler is one self-contained file

- Status: Accepted
- Date: 2026-11-21

## Context

The leaderboard downloads only the workspace's notebook file and imports it in a sandbox. That
sandbox runs only the setup cell and top-level definitions, has no network, and provides only
marimo, qiskit, numpy, scipy and the standard library. The validator's child process loads a
`file.py:function` with `python -I`, so sibling modules are not importable either.

## Decision

Each idea is `compilers/<idea>/compiler.py`, a single file importing only the standard library,
qiskit, numpy and scipy. It defines a top-level `compile_circuit(circuit, architecture_spec)`
returning `(QuantumCircuit, obj)` where `obj.initial_loading` is the loading. Shared code is copied,
not imported. Start from `compilers/naive/compiler.py`, which contains the required `Move` and
`Architecture` plumbing. `bench/package.py` wraps the file in a marimo notebook's setup cell and
rejects disallowed imports.

## Consequences

- Some code is duplicated across ideas. Merging two ideas means combining source by hand.
- No `from __future__` tricks are needed (package.py strips them), and no relative imports are
  allowed.
- The compiler must not depend on the `architecture_spec` argument being a dict (it can be `None`
  in the notebook demo cell). Read geometry from it when it's a dict, and fall back to the default
  architecture otherwise.
