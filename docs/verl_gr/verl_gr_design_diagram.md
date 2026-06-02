# `verl_gr` Design Diagram

This diagram shows how the three recipe workloads plug into the shared `verl_gr`
runtime after the recipe refactor. The main path is:

1. `verl_gr.trainers.main_ppo` selects a task runtime and builds datasets.
2. `RecipeTaskRuntime` or a recipe task prepares tokenizer, processor, worker class, and rollout registration.
3. `RLTrainer` delegates recipe-specific generation and validation through `TrainerTaskAdapter`.
4. Custom beam workloads register rollout replicas and async agent loops under `verl_gr.workers.rollout`.
   Beam expansion itself runs in rollout-server classes (engine side), while
   agent loops focus on request grouping and metadata routing.

```mermaid
classDiagram
direction LR

class Dataset
class ActorRolloutRefWorker
class ServerAdapter
class vLLMHttpServer
class vLLMReplica
class RayPPOTrainerBase
class TrainerTaskAdapter
class SingleTurnAgentLoop
class AgentLoopWorker
class AgentLoopManager

class TaskRunner {
  +run(config)
}
class TaskSpec {
  +name
  +factory
}
class TaskFactory {
  +build_task(config)
  +load_object(class_path)
}

class RecipeTaskRuntime {
  +sanitize_fsdp2_wrap_policy(config)
  +expand_rollout_counts(config)
  +configure_rollout(config)
  +get_actor_rollout_ref_worker(config)
  +prepare(config) dict
}

class OneRecTask {
  +expand_rollout_counts(config)
  +configure_rollout(config)
  +get_actor_rollout_ref_worker(config)
}
class MiniOneRecTask {
  +expand_rollout_counts(config)
  +configure_rollout(config)
  +get_actor_rollout_ref_worker(config)
}
class RankGRPOTask {
  +prepare(config) dict
}
RecipeTaskRuntime <|-- OneRecTask
RecipeTaskRuntime <|-- MiniOneRecTask
RecipeTaskRuntime <|-- RankGRPOTask

class OneRecDataset {
  +__init__(data_files, tokenizer, config, processor, max_samples)
  +__getitem__(index) dict
}
class MiniOneRecDataset {
  +__init__(data_files, tokenizer, config, processor, max_samples)
  +__getitem__(index) dict
}
class RankGRPODataset {
  +__init__(data_files, tokenizer, config, processor, max_samples)
  +__getitem__(index) dict
}
Dataset <|-- OneRecDataset
Dataset <|-- MiniOneRecDataset
Dataset <|-- RankGRPODataset

class RLTrainer {
  +compute_advantage(data, adv_estimator)
  +_get_task_adapter() TrainerTaskAdapter
  +_prepare_recommendation_gen_batch(batch) DataProto
  +_validate()
}
RayPPOTrainerBase <|-- RLTrainer

class OpenOneRecTrainerAdapter {
  +prepare_gen_batch(trainer, batch)
  +validate(trainer)
  +dump_generations(...)
}
class MiniOneRecTrainerAdapter {
  +prepare_gen_batch(trainer, batch)
  +postprocess_rewards(trainer, batch, reward_batch)
  +validate(trainer)
}
class RankGRPOTrainerAdapter {
  +prepare_gen_batch(trainer, batch)
  +validate(trainer)
}
TrainerTaskAdapter <|-- OpenOneRecTrainerAdapter
TrainerTaskAdapter <|-- MiniOneRecTrainerAdapter
TrainerTaskAdapter <|-- RankGRPOTrainerAdapter

class RankGRPOAlgorithm {
  +rankgrpo_enabled(config)
  +compute_rank_grpo_advantage(data, config, tokenizer)
}
class RankGRPOReward {
  +compute_score(...)
  +rank_rewards_from_text(...)
}
class RankGRPOTokenizer {
  +build_rankgrpo_tokenizer_and_processor(...)
}

class OneRecActorRolloutRefWorker {
  +init_model()
}
class MiniOneRecActorRolloutRefWorker {
  +init_model()
}
ActorRolloutRefWorker <|-- OneRecActorRolloutRefWorker
ActorRolloutRefWorker <|-- MiniOneRecActorRolloutRefWorker

class RolloutRegistration {
  +register_two_stage_rollout_class()
  +register_two_stage_replica()
  +register_constrained_beam_rollout_class()
  +register_constrained_beam_replica()
}
class TwoStagevLLMRollout {
  +_two_stage_generation(prompts, kwargs) DataProto
}
class ConstrainedBeamvLLMRollout {
  +update_weights(weights, global_steps)
}
ServerAdapter <|-- TwoStagevLLMRollout
ServerAdapter <|-- ConstrainedBeamvLLMRollout

class TwoStagevLLMHttpServer {
  +generate(prompt_ids, sampling_params, request_id, image_data, video_data)
}
class ConstrainedBeamvLLMHttpServer {
  +generate(prompt_ids, sampling_params, request_id, image_data, video_data)
  +abort_all_requests(reset_prefix_cache)
}
vLLMHttpServer <|-- TwoStagevLLMHttpServer
vLLMHttpServer <|-- ConstrainedBeamvLLMHttpServer

class TwoStagevLLMReplica
class ConstrainedBeamvLLMReplica
vLLMReplica <|-- TwoStagevLLMReplica
vLLMReplica <|-- ConstrainedBeamvLLMReplica
TwoStagevLLMReplica ..> TwoStagevLLMHttpServer : server class
ConstrainedBeamvLLMReplica ..> ConstrainedBeamvLLMHttpServer : server class

class BeamBackend {
  +run_async_beam_search(...)
  +beam_search_score(candidate)
}
class BeamCandidate {
  +prompt_token_ids
  +generated_token_ids
  +cumulative_logprob
}
BeamBackend ..> BeamCandidate : ranks
TwoStagevLLMHttpServer ..> BeamBackend : stage2 beams
ConstrainedBeamvLLMHttpServer ..> BeamBackend : constrained beams

class OpenOneRecTwoStageAgentLoop {
  +run(sampling_params, kwargs) AgentLoopOutput
}
class OpenOneRecAgentLoopWorker {
  +generate_sequences(batch)
}
class OpenOneRecAgentLoopManager
SingleTurnAgentLoop <|-- OpenOneRecTwoStageAgentLoop
AgentLoopWorker <|-- OpenOneRecAgentLoopWorker
AgentLoopManager <|-- OpenOneRecAgentLoopManager
OpenOneRecAgentLoopWorker ..> OpenOneRecTwoStageAgentLoop : registered loop

class MiniOneRecConstrainedBeamAgentLoop {
  +run(sampling_params, kwargs) AgentLoopOutput
}
class MiniOneRecConstrainedBeamAgentLoopWorker {
  +generate_sequences(batch)
}
class MiniOneRecConstrainedBeamAgentLoopManager
SingleTurnAgentLoop <|-- MiniOneRecConstrainedBeamAgentLoop
AgentLoopWorker <|-- MiniOneRecConstrainedBeamAgentLoopWorker
AgentLoopManager <|-- MiniOneRecConstrainedBeamAgentLoopManager
MiniOneRecConstrainedBeamAgentLoopWorker ..> MiniOneRecConstrainedBeamAgentLoop : registered loop

TaskRunner ..> TaskSpec : registry
TaskRunner ..> OneRecTask : openonerec
TaskRunner ..> RankGRPOTask : rankgrpo
TaskFactory ..> OneRecTask : default class path
TaskFactory ..> MiniOneRecTask : minionerec class path
TaskFactory ..> RecipeTaskRuntime : standalone loader
TaskRunner ..> RLTrainer : constructs
TaskRunner ..> OneRecDataset : create rl dataset
TaskRunner ..> MiniOneRecDataset : create rl dataset
TaskRunner ..> RankGRPODataset : create rl dataset

OneRecTask ..> RolloutRegistration : two stage
OneRecTask ..> OneRecActorRolloutRefWorker : selects
OneRecTask ..> OpenOneRecAgentLoopManager : configures
OneRecActorRolloutRefWorker ..> TwoStagevLLMRollout : registers

MiniOneRecTask ..> RolloutRegistration : constrained beam
MiniOneRecTask ..> MiniOneRecActorRolloutRefWorker : selects
MiniOneRecTask ..> MiniOneRecConstrainedBeamAgentLoopManager : configures
MiniOneRecActorRolloutRefWorker ..> ConstrainedBeamvLLMRollout : registers

RankGRPOTask ..> RankGRPOTokenizer : builds tokenizer
RankGRPOTask ..> ActorRolloutRefWorker : vanilla vllm
RLTrainer ..> OpenOneRecTrainerAdapter : task adapter
RLTrainer ..> RankGRPOTrainerAdapter : task adapter
RLTrainer ..> MiniOneRecTrainerAdapter : adapter extension
RLTrainer ..> RankGRPOAlgorithm : rank advantages
RankGRPOAlgorithm ..> RankGRPOReward : per rank rewards
```

## Recipe Integration Notes

- OpenOneRec uses `OneRecTask` to expand rollout counts by beam width, register the
  `two_stage` async rollout path, select `OneRecActorRolloutRefWorker`, and wire
  `OpenOneRecAgentLoopManager`. Its dataset, reward, and task runtime still live
  in `verl_gr/recipes/openonerec/onerec_recipe.py`; validation and checkpoint
  pruning live in `verl_gr/recipes/openonerec/onerec_trainer.py`.
  The two-stage beam decode is executed in
  `workers/rollout/two_stage_vllm_async.py::TwoStagevLLMHttpServer`
  (stage cache + semaphore + beam backend), not inside trainer-side Python loops.
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
  registration helpers, two async vLLM server subclasses, rollout adapter classes,
  and the shared async beam backend used by both beam-search recipes.
  This is the current performance-critical decode layer replacing the old
  high-level Python-only beam orchestration idea.

## Diagram Legend

- Solid inheritance arrows show local classes subclassing upstream `verl` bases or
  other local abstractions.
- Dotted arrows show runtime selection, registration, delegation, or dependency
  edges rather than inheritance.
- The diagram includes external upstream bases only where they make the `verl_gr`
  class relationships easier to read.
