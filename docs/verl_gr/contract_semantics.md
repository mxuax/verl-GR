# Contract Semantics

In `verl-GR`, a contract defines the stable interface between framework stages.

## What a Contract Must Define

- required inputs
- optional inputs
- outputs
- stable fields that downstream stages may rely on

## What a Contract Must Not Do

- implement backend-specific runtime logic
- embed task-specific business logic
- redefine objective behavior inside stage interfaces

## Important Distinction

- `*_contract.py` files define stage interfaces
- `objective_schema.py` defines optimization target structure
