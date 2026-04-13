# Sample Schema

Phase A defines one shared sample protocol for all task paths.

## Shared Fields

- `sample_id`
- `task_type`
- `input_context`
- `target`
- `metadata`

## Path-Specific Fields

### SID Path

- `item_history`
- `sid_target`
- `sid_delimiter`

### Natural-Language Path

- `conversation`
- `nl_target`

## Stage Minimum Fields

- SFT: `sample_id`, `task_type`, `input_context`
- Distill: `sample_id`, `task_type`, `input_context`
- RL: `sample_id`, `task_type`, `input_context`

Later phases may extend the schema, but should not remove or reinterpret these shared fields.
