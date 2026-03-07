from typing import List, Dict, Any, TypedDict, Optional, Union, Literal
from abc import ABC, abstractmethod
from dataclasses import dataclass
from skyrl_train.inference_engines.base import ConversationType

TrainingPhase = Literal["train", "eval"]


@dataclass
class TrajectoryID:
    instance_id: str  # Unique identifier for the instance in the dataset
    repetition_id: int  # Which sample/repetition for this UID (0, 1, 2... for GRPO)

    def to_string(self) -> str:
        return f"{self.instance_id}_{self.repetition_id}"


@dataclass
class BatchMetadata:
    global_step: int
    training_phase: TrainingPhase


class GeneratorInput(TypedDict):
    prompts: List[ConversationType]
    env_classes: List[str]
    env_extras: Optional[List[Dict[str, Any]]]
    sampling_params: Optional[Dict[str, Any]]
    trajectory_ids: Optional[List[TrajectoryID]]
    batch_metadata: Optional[BatchMetadata]


class GeneratorOutput(TypedDict):
    prompt_token_ids: List[List[int]]
    response_ids: List[List[int]]
    rewards: Union[List[float], List[List[float]]]
    loss_masks: List[List[int]]
    stop_reasons: Optional[List[str]]
    rollout_metrics: Optional[Dict[str, Any]]
    rollout_logprobs: Optional[List[List[float]]]
    trajectory_ids: Optional[List[TrajectoryID]]
    # Applicable only for step-wise training
    is_last_step: Optional[List[bool]]
    # R3 (Rollout Routing Replay): MoE expert routing captured during inference.
    # Each element is a numpy int32 array of shape (response_len, num_layers, top_k).
    routed_experts: Optional[List[Any]]


class MetricsOutput(TypedDict):
    avg_score: Optional[float]
    pass_at_n: Optional[float]
    mean_positive_reward: Optional[float]


class GeneratorInterface(ABC):
    @abstractmethod
    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """Generate trajectories for the input batch.

        Returns outputs in the same order as the input batch.

        Args:
            input_batch (GeneratorInput): Input batch
        Returns:
            GeneratorOutput: Generated trajectories

        Subclasses must implement this method to:
        1. Process prompts from input_batch
        2. Call inference engine to generate responses
        3. Compute rewards if applicable
        4. Return GeneratorOutput with all required fields
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.generate() must be implemented by subclasses. "
            f"This method should process input prompts, generate responses via the inference engine, "
            f"and return a GeneratorOutput dict with prompt_token_ids, response_ids, rewards, etc."
        )
