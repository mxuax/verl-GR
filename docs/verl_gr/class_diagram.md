# `verl_gr` Class Diagram

```mermaid
classDiagram
direction LR

class Dataset
class ActorRolloutRefWorker
class AsyncActorRolloutRefWorker
class ServerAdapter
class vLLMHttpServer
class vLLMReplica
class RayPPOTrainerBase
class SingleTurnAgentLoop
class AgentLoopWorker
class AgentLoopManager

class OneRecDataset {
  +__init__(data_files, tokenizer, config, processor, max_samples)
  +__getitem__(index) dict
  +maybe_filter_out_long_prompts(dataframe) Dataset
}
Dataset <|-- OneRecDataset

class OneRecTask {
  +sanitize_fsdp2_wrap_policy(config)
  +prepare(config) dict
}

class OneRecActorRolloutRefWorker {
  +_build_rollout(trust_remote_code)
  +update_weights(global_steps)
}
ActorRolloutRefWorker <|-- OneRecActorRolloutRefWorker

class OneRecAsyncActorRolloutRefWorker {
  +_build_rollout(trust_remote_code)
}
AsyncActorRolloutRefWorker <|-- OneRecAsyncActorRolloutRefWorker

class TwoStagevLLMRollout {
  +generate_sequences(prompts, kwargs) DataProto
  +_two_stage_generation(prompts, kwargs) DataProto
}
ServerAdapter <|-- TwoStagevLLMRollout

class TwoStagevLLMHttpServer {
  +generate(prompt_ids, sampling_params, request_id, image_data, video_data, priority) TokenOutput
  +_generate_two_stage(...) TokenOutput
}
vLLMHttpServer <|-- TwoStagevLLMHttpServer

class TwoStagevLLMReplica {
  +__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model, is_teacher_model)
}
vLLMReplica <|-- TwoStagevLLMReplica
TwoStagevLLMReplica --> TwoStagevLLMHttpServer : sets server_class

class RLTrainer {
  +_get_gen_batch(batch) DataProto
  +_validate()
}
RayPPOTrainerBase <|-- RLTrainer

class ValidationGenerationsLogger {
  +log(logger_backends, samples, global_step)
}
RLTrainer ..> ValidationGenerationsLogger : validation logging

class OpenOneRecTwoStageAgentLoop {
  +run(sampling_params, kwargs) AgentLoopOutput
}
SingleTurnAgentLoop <|-- OpenOneRecTwoStageAgentLoop

class OpenOneRecAgentLoopWorker {
  +generate_sequences(batch)
}
AgentLoopWorker <|-- OpenOneRecAgentLoopWorker
OpenOneRecAgentLoopWorker ..> OpenOneRecTwoStageAgentLoop : uses registered agent

class OpenOneRecAgentLoopManager {
  +agent_loop_workers_class
}
AgentLoopManager <|-- OpenOneRecAgentLoopManager
OpenOneRecAgentLoopManager ..> OpenOneRecAgentLoopWorker : remote worker class

class PreparedPromptInputs {
  +vllm_inputs: list[dict]
  +non_tensor_batch: dict
}

class CandidateExpansion {
  +responses: list[list[int]]
  +idx: torch.Tensor
  +attention_mask: torch.Tensor
  +position_ids: torch.Tensor
  +non_tensor_batch: dict
  +batch_size: int
}

TwoStagevLLMRollout ..> PreparedPromptInputs : prepare_prompt_token_inputs()
TwoStagevLLMRollout ..> CandidateExpansion : expand_beam_candidates()

OneRecTask ..> OneRecActorRolloutRefWorker : chooses worker class
OneRecTask ..> OneRecAsyncActorRolloutRefWorker : chooses async worker class
OneRecActorRolloutRefWorker ..> TwoStagevLLMRollout : builds custom rollout
```

## Notes

- Solid inheritance arrows show subclassing against either local or upstream `verl` classes.
- Dotted arrows indicate runtime dependency/selection (factory/registration/usage), not inheritance.
- The diagram focuses on classes under `verl_gr` and only includes external bases needed to make relationships readable.
