# Directory Semantics

The directory structure in Phase A is frozen around four roles:

- `recipes/`: task wiring and entrypoints
- `trainers/`: stage orchestration
- `components/`: reusable implementations
- `integrations/`: thin bridges to existing systems

## Boundary Rule

If one file appears to own more than one of these roles, the design should be
revisited before new code is added.
