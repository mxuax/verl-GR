# Layout

Phase A freezes the top-level layout for the `verl_gr` package.

## Directories

- `contracts/`: schemas, stage interfaces, and composition rules
- `recipes/`: task wiring and entrypoint assembly
- `trainers/`: stage-level orchestration skeletons
- `components/`: reusable local implementations
- `integrations/`: thin runtime bridges reserved for later phases
- `configs/`: task and stage configuration files reserved for later phases
- `docs/`: architecture and engineering notes

## Boundary Rules

- `recipe = task wiring`
- `trainer = stage orchestration`
- `component = reusable implementation`
- `integration = thin bridge`

If one file needs to own more than one of these roles, the design should be revisited before implementation continues.
