# `verl_gr` Design Diagram

This diagram shows how the three recipe workloads plug into the shared `verl_gr`
runtime after the recipe refactor. The main path is:

1. `verl_gr.trainers.main_ppo` selects a task runtime and builds datasets.
2. `RecipeTaskRuntime` or a recipe task prepares tokenizer, processor, worker class, and rollout registration.
3. `RLTrainer` delegates recipe-specific generation and validation through `TrainerTaskAdapter`.
4. Custom beam workloads register rollout replicas and async agent loops under `verl_gr.workers.rollout`.
   **OpenOneRec** and **MiniOneRec async** paths run beam expansion in rollout-server
   classes (`TwoStagevLLMHttpServer`, `ConstrainedBeamvLLMHttpServer`); agent loops only
   group requests and attach metadata (`beam_index`, `beam_group_id`, etc.). MiniOneRec
   DDP training additionally routes to `hf_constrained_beam_generate` on the actor worker.

```mermaid
flowchart LR
  %% High-level runtime ownership. Agent loops route requests; beam search runs
  %% in rollout/engine implementations except for MiniOneRec's DDP HF path.

  subgraph Entry["Entrypoint and Task Runtime"]
    Main["main_ppo.TaskRunner"]
    Factory["task_factory / task registry"]
    Runtime["RecipeTaskRuntime"]
    OpenTask["OneRecTask"]
    MiniTask["MiniOneRecTask"]
    RankTask["RankGRPOTask"]
    Data["Recipe datasets"]
  end

  subgraph Trainer["Shared Trainer Layer"]
    RLTrainer["RLTrainer"]
    Adapter["TrainerTaskAdapter"]
    OpenAdapter["OpenOneRecTrainerAdapter"]
    MiniAdapter["MiniOneRecTrainerAdapter"]
    RankAdapter["RankGRPOTrainerAdapter"]
    RankAlgo["RankGRPO advantage / reward"]
  end

  subgraph OpenRollout["OpenOneRec Two-Stage Rollout"]
    OpenWorker["OneRecActorRolloutRefWorker"]
    TwoStageRollout["TwoStagevLLMRollout"]
    OpenManager["OpenOneRecAgentLoopManager"]
    OpenAgent["OpenOneRecTwoStageAgentLoop\nmetadata only"]
    TwoStageServer["TwoStagevLLMHttpServer\nstage-1 sample + stage-2 beam"]
  end

  subgraph MiniRollout["MiniOneRec Constrained Rollout"]
    MiniWorker["MiniOneRecActorRolloutRefWorker"]
    MiniManager["MiniOneRecConstrainedBeamAgentLoopManager"]
    HFGen["HfConstrainedBeamGenerator\nDDP / HF generate path"]
    MiniAgent["MiniOneRecConstrainedBeamAgentLoop\nasync metadata only"]
    ConstrainedRollout["ConstrainedBeamvLLMRollout\noptional async path"]
    ConstrainedServer["ConstrainedBeamvLLMHttpServer\nconstrained beam"]
  end

  subgraph BeamInfra["Reusable Rollout Engine Infrastructure"]
    Registration["rollout.registration"]
    BeamBackend["beam_backend.run_async_beam_search"]
    vLLMServer["vLLMHttpServer / vLLMReplica"]
  end

  Main --> Factory
  Factory --> Runtime
  Runtime --> OpenTask
  Runtime --> MiniTask
  Runtime --> RankTask
  Main --> Data
  Main --> RLTrainer

  RLTrainer --> Adapter
  Adapter --> OpenAdapter
  Adapter --> MiniAdapter
  Adapter --> RankAdapter
  RLTrainer --> RankAlgo
  RankTask --> RankAlgo

  OpenTask -->|registers two_stage| Registration
  OpenTask --> OpenWorker
  OpenTask --> OpenManager
  OpenWorker -->|registers rollout engine| TwoStageRollout
  TwoStageRollout -->|dispatches async generation| TwoStageServer
  OpenManager --> OpenAgent
  OpenAgent -->|server_manager.generate| TwoStageServer
  TwoStageServer -->|stage-2 beam expansion| BeamBackend

  MiniTask -->|registers constrained_beam| Registration
  MiniTask --> MiniWorker
  MiniTask --> MiniManager
  MiniManager -->|default DDP path| MiniWorker
  MiniWorker -->|hf_constrained_beam_generate| HFGen
  MiniWorker -.->|optional async registration| ConstrainedRollout
  MiniManager -.->|fallback async path| MiniAgent
  MiniAgent -.->|server_manager.generate| ConstrainedServer
  ConstrainedRollout -.-> ConstrainedServer
  ConstrainedServer -.->|constrained beam expansion| BeamBackend

  Registration --> vLLMServer
  TwoStageServer --> vLLMServer
  ConstrainedServer --> vLLMServer
```

## Recipe Integration Notes

- OpenOneRec uses `OneRecTask` to expand rollout counts by beam width, register the
  `two_stage` async rollout path, select `OneRecActorRolloutRefWorker`, and wire
  `OpenOneRecAgentLoopManager`. Dataset, reward, and task runtime live in
  `verl_gr/recipes/openonerec/onerec_recipe.py`; validation and checkpoint pruning
  live in `onerec_trainer.py`.
  **Decode path (engine / rollout layer, not trainer):**
  - `OpenOneRecTwoStageAgentLoop` / `OpenOneRecAgentLoopWorker` only attach
    `stage1_sample_idx`, `beam_index`, and `beam_group_id`, then call
    `server_manager.generate(...)`.
  - Stage-1 sampling, stage-2 beam expansion, per-group cache reuse, and inflight
    throttling run in `workers/rollout/two_stage_vllm_async.py::TwoStagevLLMHttpServer`
    (`_generate_two_stage`, `_build_two_stage_cache_entry`, `_run_stage2_beam_search`).
  - Shared beam kernel: `workers/rollout/beam_backend.py::run_async_beam_search`.
- MiniOneRec uses `MiniOneRecTask` to register `constrained_beam`, select
  `MiniOneRecActorRolloutRefWorker`, and wire `MiniOneRecConstrainedBeamAgentLoopManager`.
  Dataset, reward, format helpers, worker shim, agent loop, and trainer adapter
  are separate recipe modules under `verl_gr/recipes/minionerec`.
  For DDP-aligned training, MiniOneRec agent loop routes generation to
  `hf_constrained_beam_generate` on worker side (HF `model.generate()` path);
  async constrained-vLLM remains available via rollout-server classes.
- Rank-GRPO keeps rollout on the upstream vanilla `vllm` path. Its recipe code is
  now split across `rankgrpo_dataset.py`, `rankgrpo_task.py`,
  `rankgrpo_algorithm.py`, `rankgrpo_trainer.py`, `rankgrpo_reward.py`, and
  `rankgrpo_tokenizer.py`, while `rankgrpo_recipe.py` remains a compatibility
  export module for existing config overrides.

## Shared Runtime Flow

- `main_ppo.TaskRunner` resolves a task runtime, calls `prepare(config)`, creates
  train and validation datasets through upstream `create_rl_dataset`, then builds
  `RLTrainer`. Its current in-file registry directly maps `openonerec` and
  `rankgrpo`; `task_factory.py` remains the class-path loader for config-driven
  recipe task construction such as MiniOneRec.
- `RecipeTaskRuntime` centralizes common FSDP wrap-policy cleanup, HuggingFace
  tokenizer/processor creation, worker class selection, and rollout configuration
  hooks. OpenOneRec and MiniOneRec override the rollout hooks; Rank-GRPO overrides
  `prepare` to keep its tokenizer behavior unchanged.
- `RLTrainer` owns shared recommendation generation batch preparation. It delegates
  recipe validation and generation logging through task adapters, and calls
  `rankgrpo_algorithm.compute_rank_grpo_advantage` only when
  `algorithm.rank_grpo.enable` is true.
- `verl_gr.workers.rollout` contains the reusable beam-search infrastructure:
  registration helpers, async vLLM server subclasses (`TwoStagevLLMHttpServer`,
  `ConstrainedBeamvLLMHttpServer`), rollout adapter classes, and
  `beam_backend.run_async_beam_search`. OpenOneRec and MiniOneRec async paths
  execute beam expansion in these rollout-server classes; MiniOneRec DDP training
  additionally uses `HfConstrainedBeamGenerator` on the actor worker.
  This engine-side decode layer replaces the earlier trainer-side Python beam idea.

## Diagram Legend

- Solid arrows show the primary runtime path.
- Dotted arrows show optional or fallback paths, mainly MiniOneRec's async vLLM route.
- Box groups show ownership boundaries: task/runtime selection, shared trainer,
  recipe-specific rollout wiring, and reusable rollout-engine infrastructure.
