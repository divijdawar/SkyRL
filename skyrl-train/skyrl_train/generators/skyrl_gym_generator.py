"""
This file implements ``SkyRLGymGenerator``, an implementation of the `GeneratorInterface` that
uses SkyRL-Gym as the environment.

For details, see https://skyrl.readthedocs.io/en/latest/tutorials/skyrl_gym_generator.html
"""

import asyncio
import copy
from uuid import uuid4
import skyrl_gym
from typing import List, Dict, Any, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
from tqdm.asyncio import tqdm
from dataclasses import dataclass
from loguru import logger

from skyrl_train.generators.base import GeneratorInterface, GeneratorInput, GeneratorOutput, TrajectoryID
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.base import InferenceEngineInput, ConversationType
from omegaconf import DictConfig
from skyrl_gym.envs.base_text_env import BaseTextEnvStepOutput
from skyrl_train.generators.utils import (
    get_custom_chat_template,
    get_generation_prompt_ids,
    apply_overlong_filtering,
    get_rollout_metrics,
)


@dataclass
class TrajectoryOutput:
    """Output from a single agent_loop execution."""

    response_ids: List[int]
    reward: Union[List[float], float]
    stop_reason: str
    loss_mask: List[int]
    prompt_ids: List[int]
    rollout_logprobs: Optional[List[float]]
    env_metrics: Dict[str, Any]
    routed_experts: Optional[Any] = None  # numpy int32 array (response_len, num_layers, top_k)


@dataclass
class StepWiseOutput:
    """Output from a single agent_loop execution for step-wise training."""

    step_outputs: List[TrajectoryOutput]


@dataclass
class AgentLoopState:
    chat_history: ConversationType
    input_ids: List[int]
    loss_mask: List[int]
    rollout_logprobs: Optional[List[float]]
    response_end_idx: Optional[int]
    done: bool


@dataclass
class TurnOutput:
    output: str
    output_ids: List[int]
    output_logprobs: Optional[List[float]]
    new_obs: ConversationType
    obs_ids: List[int]
    reward: Optional[float]
    added_eos: bool = False
    routed_experts: Optional[Any] = None  # numpy int32 array (output_len, num_layers, top_k)

    def get_turn_loss_mask(self) -> List[int]:
        """
        Get loss mask for this turn's tokens.

        Returns:
            List[int]: Loss mask where 1 indicates tokens to include in loss computation and 0 indicates
                tokens to exclude. If `added_eos` is True, the EOS token is masked out (set to 0).
                Observation tokens are always masked out (set to 0).
        """
        # if `added_eos` is `True`, then  the EOS token was not generated and only added in the
        # `agent_loop` function. For consistency with other entities like logprobs , we ignore it in the loss
        # mask
        return ([1] * len(self.output_ids) if not self.added_eos else [1] * (len(self.output_ids) - 1) + [0]) + [
            0
        ] * len(self.obs_ids)

    def get_turn_rollout_logprobs(self) -> Optional[List[float]]:
        """
        Get rollout logprobs for this turn's tokens.

        Returns:
            Optional[List[float]]: Logprobs for output tokens followed by dummy values (0.0) for
                observation tokens. Returns None if output_logprobs is None.
        """
        if not self.output_logprobs:
            return None
        return self.output_logprobs + [0.0] * len(self.obs_ids)


class SkyRLGymGenerator(GeneratorInterface):
    def __init__(
        self,
        generator_cfg: DictConfig,
        skyrl_gym_cfg: DictConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
        model_name: str,
    ):
        """
        Args:
            generator_cfg: DictConfig object containing the generator configuration
            inference_engine_client: InferenceEngineClient object for interacting with the inference engines
            tokenizer: tokenizer object for encoding and decoding text
        """
        self.generator_cfg = generator_cfg
        self.skyrl_gym_cfg = skyrl_gym_cfg
        self.inference_engine_client = inference_engine_client
        self.tokenizer = tokenizer
        self.max_turns = generator_cfg.max_turns
        self.batched = generator_cfg.batched
        self.use_conversation_multi_turn = generator_cfg.use_conversation_multi_turn
        # optionally use custom chat template to get loss masks (i.e. for Qwen3)
        self.custom_chat_template = get_custom_chat_template(generator_cfg.chat_template)
        # get generation prompt ids for the tokenizer if needed
        self.generation_prompt_ids = get_generation_prompt_ids(tokenizer) if self.use_conversation_multi_turn else None
        if self.skyrl_gym_cfg.max_env_workers > 0:
            self.env_executor = ThreadPoolExecutor(
                max_workers=self.skyrl_gym_cfg.max_env_workers, thread_name_prefix="skyrl-gym-env-"
            )
        else:
            self.env_executor = None

        # Session-based generation config (for multi-turn KV cache reuse)
        sessions_config = generator_cfg.get("sessions", None)
        self.sessions_enabled = sessions_config is not None and sessions_config.get("enabled", False)
        self.session_capacity = sessions_config.get("default_capacity", 8192) if sessions_config else 8192

        # Session pooling configuration
        self.pool_sessions = sessions_config.get("pool_sessions", False) if sessions_config else False
        self.max_pool_size = sessions_config.get("max_pool_size", 64) if sessions_config else 64

        # Session pool: stores available pre-opened sessions for reuse
        # When pool_sessions=True, sessions are kept open and recycled instead of closed
        self._session_pool: asyncio.Queue[str] = asyncio.Queue(maxsize=self.max_pool_size)
        self._session_pool_initialized = False
        self._session_pool_lock = asyncio.Lock()

        self._validate_cfg(generator_cfg)

        # base_conversation is used when `use_conversation_multi_turn==True and custom_chat_template==None` to
        # correctly format and tokenize observations into `observation_ids`.
        # Follows https://jybsuper.github.io/posts/multiturn_tokenization/#the-breakthrough-fixed-base-approach
        self.base_conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "I am a user."},
        ]
        self.base_conversation_token_ids = tokenizer.apply_chat_template(
            self.base_conversation,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=False,
            **self.generator_cfg.chat_template_kwargs,
        )
        # We remove tokens after the last EOS token so that it can be captured in `observation_ids`.
        # For details, see https://skyrl.readthedocs.io/en/latest/tutorials/skyrl_gym_generator.html#multi-turn-tokenization-and-ti-to
        if self.tokenizer.eos_token_id in self.base_conversation_token_ids:
            last_eos_token_index = (
                len(self.base_conversation_token_ids)
                - 1
                - self.base_conversation_token_ids[::-1].index(self.tokenizer.eos_token_id)
            )
            self.base_conversation_token_ids = self.base_conversation_token_ids[: last_eos_token_index + 1]

    def _validate_cfg(self, generator_cfg: DictConfig):
        if len(generator_cfg.chat_template_kwargs) and generator_cfg.batched:
            raise ValueError(
                "`chat_template_kwargs` is not compatible with `batched=True` since the chat templating is handled by the inference engine"
            )

        if self.generator_cfg.step_wise_trajectories:
            # Allow batched=True only when batched_multi_turn is also enabled
            if self.batched and not self.generator_cfg.get("batched_multi_turn", False):
                raise ValueError(
                    "`step_wise_trajectories` requires `batched_multi_turn=True` when `batched=True`. "
                    "Set `generator.batched_multi_turn=True` to enable batched multi-turn generation."
                )

            if self.custom_chat_template is not None:
                raise ValueError(
                    f"`step_wise_trajectories` doesn't support custom chat template, got {generator_cfg.chat_template}"
                )

            if not self.use_conversation_multi_turn:
                raise ValueError("`step_wise_trajectories` doesn't support `use_conversation_multi_turn=False`")

    async def _run_in_executor_if_available(self, func, *args, **kwargs):
        if (executor := self.env_executor) is not None:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(executor, func, *args, **kwargs)
        else:
            return func(*args, **kwargs)

    async def _initialize_session_pool(self) -> None:
        """Initialize the session pool by pre-opening sessions.

        This is called lazily on first use when pool_sessions=True.
        Pre-opens up to max_pool_size sessions for reuse.
        """
        async with self._session_pool_lock:
            if self._session_pool_initialized:
                return

            if not self.inference_engine_client.supports_sessions():
                logger.warning("Session pooling enabled but engine doesn't support sessions")
                self._session_pool_initialized = True
                return

            # Pre-open sessions up to max_pool_size
            num_to_create = min(self.max_pool_size, 16)  # Start with smaller batch
            created = 0
            for i in range(num_to_create):
                session_id = await self.inference_engine_client.open_session(
                    capacity_of_str_len=self.session_capacity,
                    session_id=f"pool_session_{i}",
                )
                if session_id:
                    try:
                        self._session_pool.put_nowait(session_id)
                        created += 1
                    except asyncio.QueueFull:
                        # Pool is full, close this session
                        await self.inference_engine_client.close_session(session_id)
                        break

            logger.info(f"Initialized session pool with {created} sessions (capacity={self.session_capacity})")
            self._session_pool_initialized = True

    async def _acquire_session(self) -> Optional[str]:
        """Acquire a session from the pool or create a new one.

        Returns:
            Session ID if successful, None otherwise.
        """
        if self.pool_sessions:
            # Lazy initialization of pool
            if not self._session_pool_initialized:
                await self._initialize_session_pool()

            try:
                # Try to get from pool (non-blocking)
                session_id = self._session_pool.get_nowait()
                logger.debug(f"Acquired session {session_id} from pool")
                return session_id
            except asyncio.QueueEmpty:
                # Pool empty, create new session
                session_id = await self.inference_engine_client.open_session(
                    capacity_of_str_len=self.session_capacity,
                )
                if session_id:
                    logger.debug(f"Created new session {session_id} (pool was empty)")
                return session_id
        else:
            # No pooling, create new session each time
            return await self.inference_engine_client.open_session(
                capacity_of_str_len=self.session_capacity,
            )

    async def _release_session(self, session_id: str) -> None:
        """Release a session back to the pool or close it.

        Args:
            session_id: The session to release.
        """
        if self.pool_sessions:
            try:
                # Try to return to pool (non-blocking)
                self._session_pool.put_nowait(session_id)
                logger.debug(f"Returned session {session_id} to pool")
            except asyncio.QueueFull:
                # Pool is full, close this session
                await self.inference_engine_client.close_session(session_id)
                logger.debug(f"Closed session {session_id} (pool full)")
        else:
            # No pooling, close session
            await self.inference_engine_client.close_session(session_id)
            logger.debug(f"Closed session {session_id}")

    async def cleanup_session_pool(self) -> None:
        """Clean up all sessions in the pool.

        Call this during shutdown to properly release all resources.
        """
        if not self.pool_sessions or not self._session_pool_initialized:
            return

        closed = 0
        while not self._session_pool.empty():
            try:
                session_id = self._session_pool.get_nowait()
                await self.inference_engine_client.close_session(session_id)
                closed += 1
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                logger.warning(f"Error closing pooled session: {e}")

        logger.info(f"Cleaned up {closed} pooled sessions")
        self._session_pool_initialized = False

    async def teardown(self) -> None:
        """Clean up resources when the generator is done.

        This should be called during trainer shutdown to release:
        - Pooled sessions (if session pooling enabled)
        - Thread pool executor (if max_env_workers > 0)
        """
        # Clean up session pool
        await self.cleanup_session_pool()

        # Shutdown thread pool executor
        if self.env_executor is not None:
            self.env_executor.shutdown(wait=False)
            logger.debug("Shutdown environment thread pool executor")

    def __del__(self):
        """Destructor to clean up resources on garbage collection.

        Note: For proper cleanup, prefer calling teardown() explicitly.
        """
        if self.env_executor is not None:
            self.env_executor.shutdown(wait=False)

    async def agent_loop(
        self,
        prompt: ConversationType,
        env_class: str,
        env_extras: Dict[str, Any],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Optional[Dict[str, Any]] = None,
        trajectory_id: Optional[TrajectoryID] = None,
    ) -> Union[TrajectoryOutput, StepWiseOutput]:
        """
        Multi-turn generation loop that executes a single trajectory.

        Note:
            We ensure token-in-token-out generation. With two exceptions:
            - When calling Env.step() and BaseTextEnvStepOutput["postprocessed_action"] is not None.
              This will likely be deprecated soon.
            - When custom_chat_template = True and use_conversation_multi_turn = True. We always
              re-tokenize the entire chat history every turn and at the end. This is used for cases
              like removing Qwen3 thinking tokens in non-last-round assistant message.

        Args:
            prompt: ConversationType
            env_extras: Dict[str, Any]
            max_tokens: int
            max_input_length: int
            sampling_params: Optional[Dict[str, Any]]
        Returns:
            response_ids: List[int]
            reward: Union[float, List[float]]
            stop_reason: str
            loss_mask: List[int]
            prompt_token_ids: List[int]
            rollout_logprobs: Optional[List[float]]
        """
        # NOTE: `custom_chat_template` was mainly for getting accurate loss masks for thinking models.
        # This is no longer needed now given that step wise training is supported
        # TODO (sumanthrh): This path can be deprecated
        retokenize_chat_history = self.use_conversation_multi_turn and self.custom_chat_template

        # Create a new environment instance
        env_extras["max_turns"] = self.max_turns  # TODO(shu): move this to config
        env_config = self.skyrl_gym_cfg.get(env_class, DictConfig({}))
        env = skyrl_gym.make(env_class, env_config=env_config, extras=env_extras)

        session_id = (
            f"{trajectory_id.instance_id}_{trajectory_id.repetition_id}" if trajectory_id is not None else uuid4().hex
        )

        # Session management: open/acquire session if enabled and supported
        use_session_api = (
            self.sessions_enabled
            and self.inference_engine_client.supports_sessions()
            and self.use_conversation_multi_turn  # Sessions only make sense for multi-turn
        )
        current_request_id: Optional[str] = None  # Track request ID for session continuity
        opened_session_id: Optional[str] = None

        if use_session_api:
            # Use session pool if enabled, otherwise create new session
            if self.pool_sessions:
                opened_session_id = await self._acquire_session()
            else:
                opened_session_id = await self.inference_engine_client.open_session(
                    capacity_of_str_len=self.session_capacity,
                    session_id=session_id,
                )

            if opened_session_id is None:
                logger.warning(f"Failed to acquire session for {session_id}, falling back to regular generation")
                use_session_api = False
            else:
                session_id = opened_session_id  # Use the actual session ID (may differ for pooled sessions)
                logger.debug(f"Acquired session {session_id} for trajectory (pooled={self.pool_sessions})")

        # Instantiate chat_history and chat_end_index, which are only used if `retokenize_chat_history==True`.
        # Need copy here since the prompt is a list of messages and we are going to modify it.
        chat_history = copy.deepcopy(prompt)

        # init() returns the first prompt to be given to the model, and optional metadata dict
        chat_history, _ = await self._run_in_executor_if_available(env.init, chat_history)
        initial_chat_history_length = len(chat_history)
        initial_input_ids = self.tokenizer.apply_chat_template(
            chat_history,
            # If retokenize_chat_history==True, avoid including the generation prompt in both the
            # prompt_ids and response_ids due to how `response_encodings["input_ids"]` works.
            add_generation_prompt=not retokenize_chat_history,
            chat_template=self.custom_chat_template if retokenize_chat_history else None,
            tokenize=True,
            return_dict=False,
            **self.generator_cfg.chat_template_kwargs,
        )

        initial_prompt_length = len(initial_input_ids)
        loss_mask = []  # this excludes the prompt
        rollout_logprobs = None

        current_sampling_params = sampling_params if sampling_params is not None else self.generator_cfg.sampling_params

        # Accumulate per-step rewards. Format: (reward, response_end_token_idx)
        per_step_rewards: List[Tuple[float, Optional[int]]] = []

        is_step_wise = self.generator_cfg.step_wise_trajectories

        agent_loop_output = StepWiseOutput(step_outputs=[]) if is_step_wise else None

        get_logprobs = self.generator_cfg.sampling_params.logprobs is not None
        agent_loop_state = AgentLoopState(
            chat_history=chat_history,
            input_ids=initial_input_ids,
            loss_mask=[],
            rollout_logprobs=[] if get_logprobs else None,
            response_end_idx=None,
            done=False,
        )

        # R3: collect per-turn routed expert arrays; concatenated and trimmed at final assembly
        r3_turn_experts: list = []

        while not agent_loop_state.done:

            if len(agent_loop_state.input_ids) > max_input_length:
                stop_reason = "length"
                break

            # 1. Generate output
            if is_step_wise or retokenize_chat_history:
                # re-apply whole chat template so length check is correct
                agent_loop_state.input_ids = self.tokenizer.apply_chat_template(
                    chat_history,
                    chat_template=self.custom_chat_template if retokenize_chat_history else None,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=False,
                    **self.generator_cfg.chat_template_kwargs,
                )
                agent_loop_state.loss_mask = []
                agent_loop_state.rollout_logprobs = None

            engine_input = InferenceEngineInput(
                prompt_token_ids=[agent_loop_state.input_ids], session_ids=[session_id], sampling_params=sampling_params
            )

            # Use session-based generation if enabled, otherwise regular generate
            if use_session_api:
                engine_output = await self.inference_engine_client.generate_with_session(
                    session_id=session_id,
                    input_batch=engine_input,
                    rid=current_request_id,  # None for first turn, previous rid for subsequent
                    drop_previous_output=True,  # Don't include previous output in new response
                )
                # Update request ID for next turn (for KV cache continuity)
                request_ids = engine_output.get("request_ids")
                if request_ids and len(request_ids) > 0:
                    current_request_id = request_ids[0]
            else:
                engine_output = await self.inference_engine_client.generate(engine_input)

            output = engine_output["responses"][0]
            output_ids = engine_output["response_ids"][0]
            stop_reason = engine_output["stop_reasons"][0]
            response_logprobs = engine_output.get("response_logprobs", None)
            if response_logprobs is not None:
                response_logprobs = response_logprobs[0]
                if self.custom_chat_template is not None:
                    raise ValueError("Response Logprobs bookkeeping is not supported with custom chat template")

            # Extract routed experts for R3 (Rollout Routing Replay)
            turn_routed_experts = engine_output.get("routed_experts", None)
            if turn_routed_experts is not None:
                turn_routed_experts = turn_routed_experts[0]

            # Append eos when sampling_params.stop is not None. Does not affect 3.a as chat templates add eos_token.
            # sampling_params is not None for eval, but None for training (which uses engine.sampling_params which are from cfg)
            stop_strs = current_sampling_params.get("stop", None)
            added_eos = False
            if (
                stop_strs is not None
                and self.generator_cfg.append_eos_token_after_stop_str_in_multi_turn
                and self.use_conversation_multi_turn
            ):
                if output.endswith(tuple(stop_strs)) and output_ids[-1] != self.tokenizer.eos_token_id:
                    output_ids.append(self.tokenizer.eos_token_id)
                    # dummy logprobs for EOS token id. It will be loss masked with 0 in TurnOutput.get_turn_loss_mask
                    if response_logprobs is not None:
                        response_logprobs.append(0.0)
                    # Pad routed experts with -1 for the appended EOS token (no routing decision)
                    if turn_routed_experts is not None:
                        import numpy as np

                        eos_pad = np.full((1, *turn_routed_experts.shape[1:]), -1, dtype=np.int32)
                        turn_routed_experts = np.concatenate([turn_routed_experts, eos_pad], axis=0)
                    added_eos = True

            # 2. Environment step
            env_step_output: BaseTextEnvStepOutput = await self._run_in_executor_if_available(env.step, output)
            new_obs = env_step_output["observations"]
            step_reward: float = env_step_output["reward"]
            agent_loop_state.done = env_step_output["done"]

            if env_step_output.get("postprocessed_action", None) is not None:
                # TODO(Charlie): come back to this, we should deprecate postprocessed action
                logger.warning(
                    "WARNING: postprocessed action may violate token-in-token-out. Ideally you "
                    "post-process it in the token space rather than string space. "
                    "A better solution coming soon."
                )
                output = env_step_output["postprocessed_action"]
                output_ids = self.tokenizer.encode(output, add_special_tokens=False)

            obs_ids = self.get_obs_ids_from_obs(new_obs, agent_loop_state.done)

            # final turn output containing generated response and environment observations
            turn_output = TurnOutput(
                output=output,
                output_ids=output_ids,
                output_logprobs=response_logprobs,
                new_obs=new_obs,
                reward=step_reward,
                obs_ids=obs_ids,
                added_eos=added_eos,
                routed_experts=turn_routed_experts,
            )

            if is_step_wise:
                # current response + observation ids
                turn_response_ids = turn_output.output_ids + turn_output.obs_ids
                turn_prompt_ids = agent_loop_state.input_ids

                # agent loop only tracks loss mask and rollout logprobs for this turn with step_wise training
                turn_loss_mask = turn_output.get_turn_loss_mask()
                turn_response_logprobs: Optional[List[float]] = turn_output.get_turn_rollout_logprobs()

                per_step_output = TrajectoryOutput(
                    response_ids=turn_response_ids,
                    reward=step_reward,
                    loss_mask=turn_loss_mask,
                    prompt_ids=turn_prompt_ids,
                    rollout_logprobs=turn_response_logprobs,
                    stop_reason=stop_reason,
                    env_metrics=env.get_metrics() if agent_loop_state.done else {},
                    routed_experts=turn_output.routed_experts,
                )
                agent_loop_output.step_outputs.append(per_step_output)

            # 3. Update states: input ids, loss_mask, chat_history, etc.
            # Three ways of managing input
            if retokenize_chat_history:
                # a. custom chat template
                agent_loop_state = self._update_agent_state_by_retokenizing_chat_history(agent_loop_state, turn_output)
            elif self.use_conversation_multi_turn:
                # b. Token-in-token-out. Follow multi-turn chat history format.
                agent_loop_state = self._update_agent_loop_state_with_multiturn_chat_template(
                    agent_loop_state, turn_output
                )
            else:
                # c. Token-in-token-out. All steps/observations are appended to a single assistant message.
                agent_loop_state = self._update_agent_loop_state_with_singleturn_chat_template(
                    agent_loop_state, turn_output
                )

            # R3: accumulate this turn's routed experts with obs padding for alignment
            if turn_output.routed_experts is not None and not is_step_wise:
                import numpy as np

                obs_pad = np.full((len(turn_output.obs_ids), *turn_output.routed_experts.shape[1:]), -1, dtype=np.int32)
                r3_turn_experts.append(np.concatenate([turn_output.routed_experts, obs_pad], axis=0))

            per_step_rewards.append((step_reward, agent_loop_state.response_end_idx))

        # Get environment-specific metrics after the episode is done
        env_metrics = env.get_metrics()
        # Close the environment
        await self._run_in_executor_if_available(env.close)

        # Release session if we opened one (returns to pool if pooling enabled)
        if use_session_api and opened_session_id is not None:
            try:
                await self._release_session(session_id)
            except Exception as e:
                logger.warning(f"Failed to release session {session_id}: {e}")

        prompt_ids = agent_loop_state.input_ids[:initial_prompt_length]
        rollout_logprobs = None
        routed_experts = None
        response_ids = None

        # Prepare the final loss_mask, response_ids and rollout_logprobs .
        # We remove the final observation messages /token IDs here
        # Note that during the agent loop, we still add the final observation messages/ tokens because we terminate the agent loop if the input length
        # exceeds the maximum
        if retokenize_chat_history:
            response_encodings = self.tokenizer.apply_chat_template(
                agent_loop_state.chat_history[
                    initial_chat_history_length : len(agent_loop_state.chat_history) - len(new_obs)
                ],
                chat_template=self.custom_chat_template,
                add_generation_prompt=False,
                return_dict=True,
                return_assistant_tokens_mask=True,
                tokenize=True,
                **self.generator_cfg.chat_template_kwargs,
            )
            loss_mask = response_encodings["assistant_masks"]
            response_ids = response_encodings["input_ids"]
        elif not self.generator_cfg.step_wise_trajectories:
            assert not any(
                agent_loop_state.loss_mask[agent_loop_state.response_end_idx - initial_prompt_length + 1 :]
            ), "loss_mask at index after response end should be all 0"
            loss_mask = agent_loop_state.loss_mask[: agent_loop_state.response_end_idx - initial_prompt_length + 1]
            response_ids = agent_loop_state.input_ids[initial_prompt_length : agent_loop_state.response_end_idx + 1]
            if agent_loop_state.rollout_logprobs is not None:
                rollout_logprobs = agent_loop_state.rollout_logprobs[
                    : agent_loop_state.response_end_idx - initial_prompt_length + 1
                ]
            # R3: concatenate per-turn experts and trim to response length
            if r3_turn_experts:
                import numpy as np

                resp_len = agent_loop_state.response_end_idx - initial_prompt_length + 1
                routed_experts = np.concatenate(r3_turn_experts, axis=0)[:resp_len]
            # fix index for per_step_rewards
            per_step_rewards = [(reward, idx - initial_prompt_length) for reward, idx in per_step_rewards]
            assert len(loss_mask) == len(
                response_ids
            ), f"loss_mask and response_ids should have the same length, got {len(loss_mask)} and {len(response_ids)}"

        appended_eos_token = False
        if not self.use_conversation_multi_turn:
            assert response_ids is not None and loss_mask is not None
            if stop_reason != "length" and response_ids and response_ids[-1] != self.tokenizer.eos_token_id:
                response_ids.append(self.tokenizer.eos_token_id)
                # TODO(Charlie): this should be 0? Otherwise logprobs will be extremely off. But if it is loss
                # masked with 0, why bother adding it?
                loss_mask.append(1)
                if rollout_logprobs is not None:
                    rollout_logprobs.append(0.0)
                # R3: pad routed experts for the appended EOS token
                if routed_experts is not None:
                    import numpy as np

                    eos_pad = np.full((1, *routed_experts.shape[1:]), -1, dtype=np.int32)
                    routed_experts = np.concatenate([routed_experts, eos_pad], axis=0)
                appended_eos_token = True

        if self.generator_cfg.step_wise_trajectories:
            for per_step_output, (reward, resp_end_idx) in zip(agent_loop_output.step_outputs, per_step_rewards):
                per_token_reward = [0.0] * len(per_step_output.response_ids)
                per_token_reward[resp_end_idx] = float(reward)
                # in-place update to per-token reward
                per_step_output.reward = per_token_reward
        else:
            reward_out = self._build_per_token_rewards(per_step_rewards, response_ids, appended_eos_token)

            agent_loop_output = TrajectoryOutput(
                response_ids=response_ids,
                reward=reward_out,
                stop_reason=stop_reason,
                loss_mask=loss_mask,
                prompt_ids=prompt_ids,
                rollout_logprobs=rollout_logprobs,
                env_metrics=env_metrics,
                routed_experts=routed_experts,
            )

        return agent_loop_output

    def _build_per_token_rewards(
        self, per_step_rewards: List[Tuple[float, Optional[int]]], response_ids: List[int], appended_eos_token: bool
    ) -> Union[float, List[float]]:
        """
        Build reward output from per-step rewards.

        Args:
            per_step_rewards: List of (reward, response_end_token_idx) tuples for each step
            response_ids: List of response token IDs
            appended_eos_token: Whether an EOS token was manually appended at the end

        Returns:
            Union[float, List[float]]: If custom_chat_template is used, returns the last step's reward (float).
                Otherwise, returns a list of token-level rewards (List[float]).
        """
        if self.custom_chat_template:
            # TODO(Charlie): Currently, the possible response truncation will not affect the reward
            # in the if branch, but some final rewards may be lost in the else branch. Fix this
            # when we support turn-level rewards for the `retokenize_chat_history` codepath.
            reward_out = per_step_rewards[-1][0]
        else:
            # Build token-level rewards placed at assistant turn boundaries
            token_level_rewards: List[float] = [0.0] * len(response_ids)
            for i, (step_reward, idx) in enumerate(per_step_rewards):
                assert step_reward is not None
                if idx >= len(response_ids):
                    break
                if appended_eos_token and i == len(per_step_rewards) - 1:
                    # NOTE(Charlie): If we appended the eos token, we need to place
                    # the reward at the last token (the manually appended eos token)
                    # rather than the last turn's assistant-generated token. This matches
                    # the logic in trainer.py::postprocess_generator_output when rewards are List[float].
                    token_level_rewards[-1] = step_reward
                else:
                    token_level_rewards[idx] += step_reward
            reward_out = token_level_rewards
        return reward_out

    def get_obs_ids_from_obs(self, new_obs: ConversationType, is_done: bool) -> List[int]:
        """
        Returns observation token ids from observation messages for a turn.

        Args:
            new_obs: Observation messages from the environment step
            is_done: Whether the agent loop has terminated

        Returns:
            List[int]: Observation token IDs. For multi-turn mode, includes chat template formatting.
                For single-turn mode, returns directly encoded observation tokens.
        """
        if self.use_conversation_multi_turn:
            # 2. apply chat template for observations, also generate generation prompt for next turn
            obs_ids_to_add = []
            if len(new_obs) > 0:
                # For Qwen, this will generate `\n<|user|>Some observation<|im_end|>\n`. Note that the
                # first `\n` is generated since we stripped it in ``base_conversation_token_ids``.
                obs_ids_to_add = self.tokenizer.apply_chat_template(
                    [*self.base_conversation, *new_obs],
                    add_generation_prompt=not is_done,
                    tokenize=True,
                    return_dict=False,
                    **self.generator_cfg.chat_template_kwargs,
                )[len(self.base_conversation_token_ids) :]
            elif not is_done:
                obs_ids_to_add = self.generation_prompt_ids
        else:
            # Build observation token ids (encoded directly, not using chat template)
            # no generation prompt is added in this case
            obs_ids_to_add = []
            if len(new_obs) > 0:
                for obs in new_obs:
                    obs_tokens = self.tokenizer.encode(obs["content"], add_special_tokens=False)
                    obs_ids_to_add.extend(obs_tokens)
        return obs_ids_to_add

    def _update_chat_history(
        self,
        chat_history: ConversationType,
        output: str,
        new_obs: ConversationType,
    ) -> ConversationType:
        """
        Update chat history with assistant response and new observations.

        Args:
            chat_history: Current conversation history
            output: Assistant's response text
            new_obs: New observation messages from the environment

        Returns:
            ConversationType: Updated chat history with assistant response and observations appended.
                The EOS token is removed from output if present, as it will be reapplied by the chat template.
        """
        # remove eos token from end of output if it exists, since it will be reapplied by the chat template
        if output.endswith(self.tokenizer.eos_token):
            output = output[: -len(self.tokenizer.eos_token)]

        # Add assistant response to chat history
        chat_history += [{"role": "assistant", "content": output}]

        # Add observations to chat history
        if len(new_obs) > 0:
            chat_history += new_obs
        return chat_history

    async def generate_batched(
        self,
        prompts: List[ConversationType],
        env_classes: List[str],
        env_extras: List[Dict[str, Any]],
        max_tokens: int,
        sampling_params: Optional[Dict[str, Any]] = None,
    ) -> GeneratorOutput:
        """
        Single-turn batched generation (can use the synchronous offline engine)

        Args:
            prompts: List[ConversationType]
            env_classes: List[str]
            env_extras: List[Dict[str, Any]]
            max_tokens: int
            max_input_length: int --> Currently unused as we assume batched is used only for single-turn.
            sampling_params: Optional[Dict[str, Any]]
        Returns:
            GeneratorOutput
        """
        envs = []
        init_prompts = []
        for env_class, env_extra, prompt in zip(env_classes, env_extras, prompts):
            env_extra["max_turns"] = self.max_turns
            env_config = self.skyrl_gym_cfg.get(env_class, DictConfig({}))
            env = skyrl_gym.make(env_class, env_config=env_config, extras=env_extra)
            init_prompt, _ = await self._run_in_executor_if_available(env.init, prompt)
            init_prompts.append(init_prompt)
            envs.append(env)

        # For single-turn generation, we can use text-in-token-out, since we do not need to re-tokenize.
        engine_input = InferenceEngineInput(prompts=init_prompts, sampling_params=sampling_params)
        engine_output = await self.inference_engine_client.generate(engine_input)
        outputs = engine_output["responses"]
        responses = engine_output["response_ids"]
        stop_reasons = engine_output["stop_reasons"]
        logprobs = engine_output.get("response_logprobs", None)

        truncated_responses = []
        rewards = []
        loss_masks = []
        env_metrics = []
        truncated_logprobs: Optional[List[List[float]]] = [] if logprobs is not None else None

        for i, (output, response, env, env_class) in enumerate(zip(outputs, responses, envs, env_classes)):
            # step on environment and compute reward
            env_step_output: BaseTextEnvStepOutput = await self._run_in_executor_if_available(env.step, output)
            reward = env_step_output["reward"]
            rewards.append(reward)

            if len(response) > max_tokens:
                response = response[:max_tokens]
            loss_masks.append([1] * len(response))
            truncated_responses.append(response)
            if logprobs is not None:
                sample_logprobs = logprobs[i][: len(response)]
                truncated_logprobs.append(sample_logprobs)

            # Get environment-specific metrics
            env_metrics.append(env.get_metrics())
            # Close the environment
            await self._run_in_executor_if_available(env.close)

        prompt_token_ids = self.tokenizer.apply_chat_template(
            init_prompts,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
        rollout_metrics = get_rollout_metrics(responses, rewards, env_metrics, env_classes)

        if self.generator_cfg.apply_overlong_filtering:
            loss_masks = apply_overlong_filtering(loss_masks, responses, self.tokenizer.eos_token_id)

        generator_output: GeneratorOutput = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": truncated_responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": truncated_logprobs,
        }

        return generator_output

    async def generate_batched_multi_turn(
        self,
        prompts: List[ConversationType],
        env_classes: List[str],
        env_extras: List[Dict[str, Any]],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Optional[Dict[str, Any]] = None,
    ) -> GeneratorOutput:
        """Multi-turn batched generation with turn-boundary synchronization.

        This method significantly improves throughput (5-8x) by batching inference
        requests across all active samples at each turn, instead of making individual
        requests per sample.

        Key optimizations:
        - Synchronizes at turn boundaries to batch requests
        - Handles variable completion times gracefully (masks completed samples)
        - Uses SGLang's continuous batching for maximum GPU utilization

        Args:
            prompts: List of conversation prompts (one per sample)
            env_classes: List of environment class names
            env_extras: List of extra environment configuration dicts
            max_tokens: Maximum tokens to generate per turn
            max_input_length: Maximum input context length
            sampling_params: Optional sampling parameters override

        Returns:
            GeneratorOutput with batched results
        """
        batch_size = len(prompts)

        # Check if we should collect logprobs
        collect_logprobs = self.generator_cfg.sampling_params.logprobs is not None

        # Initialize all environments and agent states
        agent_states: List[AgentLoopState] = []
        envs = []

        for prompt, env_class, env_extra in zip(prompts, env_classes, env_extras):
            env_extra["max_turns"] = self.max_turns
            env_config = self.skyrl_gym_cfg.get(env_class, DictConfig({}))
            env = skyrl_gym.make(env_class, env_config=env_config, extras=env_extra)

            # Initialize environment with prompt
            chat_history = copy.deepcopy(prompt)
            chat_history, _ = await self._run_in_executor_if_available(env.init, chat_history)

            # Tokenize initial prompt
            initial_input_ids = self.tokenizer.apply_chat_template(
                chat_history,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
                **self.generator_cfg.chat_template_kwargs,
            )

            # Truncate if needed
            if len(initial_input_ids) > max_input_length:
                initial_input_ids = initial_input_ids[-max_input_length:]

            agent_states.append(AgentLoopState(
                chat_history=chat_history,
                input_ids=initial_input_ids,
                loss_mask=[],
                rollout_logprobs=[] if collect_logprobs else None,
                response_end_idx=None,
                done=False,
            ))
            envs.append(env)

        # Track active samples and accumulated outputs
        active_mask = [True] * batch_size
        all_response_ids: List[List[int]] = [[] for _ in range(batch_size)]
        all_rewards: List[float] = [0.0] * batch_size
        all_stop_reasons: List[str] = [""] * batch_size
        all_loss_masks: List[List[int]] = [[] for _ in range(batch_size)]
        all_logprobs: Optional[List[List[float]]] = [[] for _ in range(batch_size)] if collect_logprobs else None
        all_env_metrics: List[Dict[str, Any]] = [{} for _ in range(batch_size)]

        turn = 0

        # Turn-by-turn batched generation
        while any(active_mask) and turn < self.max_turns:
            turn += 1

            # Collect active sample indices for this turn
            active_indices = [i for i, active in enumerate(active_mask) if active]
            if not active_indices:
                break

            # Build BATCHED request for all active samples
            batched_input_ids = [agent_states[i].input_ids for i in active_indices]

            # Use sampling params with logprobs if collecting
            turn_sampling_params = sampling_params or {}
            if collect_logprobs and "logprobs" not in turn_sampling_params:
                turn_sampling_params = {**turn_sampling_params, "logprobs": 1}

            engine_input = InferenceEngineInput(
                prompt_token_ids=batched_input_ids,
                sampling_params=turn_sampling_params,
            )

            # SINGLE batched inference call instead of N individual calls
            engine_output = await self.inference_engine_client.generate(engine_input)

            # Process each result and step environments
            for local_idx, global_idx in enumerate(active_indices):
                output_text = engine_output["responses"][local_idx]
                output_ids = engine_output["response_ids"][local_idx]
                stop_reason = engine_output.get("stop_reasons", [""])[local_idx] if engine_output.get("stop_reasons") else ""
                turn_logprobs = engine_output.get("response_logprobs", [[]])[local_idx] if engine_output.get("response_logprobs") else None

                # Truncate if needed
                if len(output_ids) > max_tokens:
                    output_ids = output_ids[:max_tokens]
                    if turn_logprobs:
                        turn_logprobs = turn_logprobs[:max_tokens]

                # Step environment
                env_step_output: BaseTextEnvStepOutput = await self._run_in_executor_if_available(
                    envs[global_idx].step, output_text
                )

                # Accumulate response and loss mask
                all_response_ids[global_idx].extend(output_ids)
                all_loss_masks[global_idx].extend([1] * len(output_ids))
                if all_logprobs is not None and turn_logprobs:
                    all_logprobs[global_idx].extend(turn_logprobs)

                # Check if done
                is_done = env_step_output.get("done", False) or stop_reason == "stop"

                if is_done:
                    active_mask[global_idx] = False
                    agent_states[global_idx].done = True
                    all_rewards[global_idx] = env_step_output.get("reward", 0.0)
                    all_stop_reasons[global_idx] = stop_reason or "stop"
                    all_env_metrics[global_idx] = envs[global_idx].get_metrics()
                else:
                    # Continue: add observation to chat history
                    agent_states[global_idx].chat_history.append(
                        {"role": "assistant", "content": output_text}
                    )

                    observation = env_step_output.get("observations", "")
                    if observation:
                        agent_states[global_idx].chat_history.append(
                            {"role": "user", "content": observation}
                        )

                    # Re-tokenize for next turn
                    new_input_ids = self.tokenizer.apply_chat_template(
                        agent_states[global_idx].chat_history,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=False,
                        **self.generator_cfg.chat_template_kwargs,
                    )

                    # Truncate if needed
                    if len(new_input_ids) > max_input_length:
                        new_input_ids = new_input_ids[-max_input_length:]

                    agent_states[global_idx].input_ids = new_input_ids

        # Handle samples that didn't complete (hit max_turns)
        for i, active in enumerate(active_mask):
            if active:
                # Get final reward for incomplete trajectories
                env_final_output = await self._run_in_executor_if_available(
                    envs[i].step, ""  # Empty step to get final state
                )
                all_rewards[i] = env_final_output.get("reward", 0.0)
                all_stop_reasons[i] = "max_turns"
                all_env_metrics[i] = envs[i].get_metrics()

        # Close all environments
        for env in envs:
            await self._run_in_executor_if_available(env.close)

        # Get prompt token IDs for output
        prompt_token_ids = [
            self.tokenizer.apply_chat_template(
                prompts[i],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
                **self.generator_cfg.chat_template_kwargs,
            )
            for i in range(batch_size)
        ]

        # Apply overlong filtering if configured
        if self.generator_cfg.apply_overlong_filtering:
            all_loss_masks = apply_overlong_filtering(
                all_loss_masks, all_response_ids, self.tokenizer.eos_token_id
            )

        # Compute rollout metrics
        rollout_metrics = get_rollout_metrics(all_response_ids, all_rewards, all_env_metrics, env_classes)

        generator_output: GeneratorOutput = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": all_response_ids,
            "rewards": all_rewards,
            "loss_masks": all_loss_masks,
            "stop_reasons": all_stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": all_logprobs,
        }

        return generator_output

    async def generate(self, input_batch: GeneratorInput, disable_tqdm: bool = False) -> GeneratorOutput:
        """
        Generate trajectories for the input batch.

        Returns outputs in the same order as the input batch.
        Args:
            input_batch: GeneratorInput
            disable_tqdm: bool
        Returns:
            GeneratorOutput
        """
        prompts = input_batch["prompts"]
        env_classes = input_batch["env_classes"]
        env_extras = input_batch["env_extras"]
        trajectory_ids = input_batch.get("trajectory_ids", None)
        if self.generator_cfg.step_wise_trajectories:
            assert trajectory_ids is not None, "`trajectory_ids` is a required field for step wise training"
        sampling_params: Optional[dict] = input_batch.get("sampling_params", None)
        max_tokens = self.generator_cfg.sampling_params.max_generate_length
        max_input_length = self.generator_cfg.max_input_length

        if self.batched:
            # Route to batched_multi_turn for multi-turn environments with batching enabled
            if self.generator_cfg.get("batched_multi_turn", False) and self.max_turns > 1:
                return await self.generate_batched_multi_turn(
                    prompts, env_classes, env_extras, max_tokens, max_input_length, sampling_params
                )
            # Single-turn batched generation
            return await self.generate_batched(prompts, env_classes, env_extras, max_tokens, sampling_params)

        # Async agent loop to generate trajectories in parallel.
        tasks = []
        for i in range(len(prompts)):
            tasks.append(
                self.agent_loop(
                    prompts[i],
                    env_classes[i],
                    env_extras[i],
                    max_tokens,
                    max_input_length,
                    sampling_params=sampling_params,
                    trajectory_id=trajectory_ids[i] if trajectory_ids is not None else None,
                )
            )

        all_outputs = await tqdm.gather(
            *tasks,
            desc="Generating Trajectories",
            miniters=max(1, len(tasks) // 10),
            mininterval=5,
            disable=disable_tqdm,
        )

        if self.generator_cfg.step_wise_trajectories:
            responses = []
            rewards = []
            stop_reasons = []
            loss_masks = []
            prompt_token_ids = []
            env_metrics = []
            is_last_step = []
            out_trajectory_ids = []
            out_env_classes = []
            for i, output in enumerate(all_outputs):
                for j, step_output in enumerate(output.step_outputs):
                    responses.append(step_output.response_ids)
                    rewards.append(step_output.reward)
                    stop_reasons.append(step_output.stop_reason)
                    loss_masks.append(step_output.loss_mask)
                    prompt_token_ids.append(step_output.prompt_ids)
                    env_metrics.append(step_output.env_metrics)
                    is_last_step.append(j == len(output.step_outputs) - 1)
                    out_trajectory_ids.append(trajectory_ids[i])
                    out_env_classes.append(env_classes[i])
            env_classes = out_env_classes
        else:
            responses = [output.response_ids for output in all_outputs]
            rewards = [output.reward for output in all_outputs]
            stop_reasons = [output.stop_reason for output in all_outputs]
            loss_masks = [output.loss_mask for output in all_outputs]
            prompt_token_ids = [output.prompt_ids for output in all_outputs]
            env_metrics = [output.env_metrics for output in all_outputs]
            is_last_step = None
            out_trajectory_ids = None

        if sampling_params is not None:
            # sampling params will be a dict in the format of the inference engine backend
            # Check both vLLM-style "logprobs" and SGLang-style "return_logprob"
            get_logprobs = (
                sampling_params.get("logprobs", None) is not None
                or sampling_params.get("return_logprob", False)
            )
        else:
            get_logprobs = self.generator_cfg.sampling_params.logprobs is not None

        if get_logprobs:
            if self.generator_cfg.step_wise_trajectories:
                rollout_logprobs = sum(
                    [[step_output.rollout_logprobs for step_output in output.step_outputs] for output in all_outputs],
                    [],
                )
            else:
                rollout_logprobs = [output.rollout_logprobs for output in all_outputs]
        else:
            rollout_logprobs = None

        # R3: collect routed experts from trajectory outputs
        if self.generator_cfg.step_wise_trajectories:
            routed_experts_list = sum(
                [[step_output.routed_experts for step_output in output.step_outputs] for output in all_outputs],
                [],
            )
        else:
            routed_experts_list = [output.routed_experts for output in all_outputs]
        # Set to None if no trajectory had routed experts (R3 disabled or non-MoE model)
        if all(re is None for re in routed_experts_list):
            routed_experts_list = None

        rollout_metrics = get_rollout_metrics(responses, rewards, env_metrics, env_classes)

        if self.generator_cfg.zero_reward_on_non_stop:
            # set reward to 0 if the stop reason is not "stop"
            rewards = self._zero_reward_if_not_stop(rewards, stop_reasons)

        if self.generator_cfg.apply_overlong_filtering:
            loss_masks = apply_overlong_filtering(loss_masks, responses, self.tokenizer.eos_token_id)

        generator_output: GeneratorOutput = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": rollout_logprobs,
            "trajectory_ids": out_trajectory_ids,
            "is_last_step": is_last_step,
            "routed_experts": routed_experts_list,
        }

        return generator_output

    def _zero_reward_if_not_stop(
        self, rewards: List[Union[float, List[float]]], stop_reasons: List[str]
    ) -> List[Union[float, List[float]]]:
        """
        Sets the reward to 0 if the stop reason is not "stop".

        This can be useful in cases where the LLM generation was truncated or aborted, but the environment still assigns non-zero reward.
        Often, we have format rewards for the LLM to follow, but in cases where the LLM didn't finish the response,
        we typically don't want to reward it. This is a general setting for all environments.

        Args:
            rewards: List of rewards (can be float or List[float] for per-token rewards)
            stop_reasons: List of stop reasons for each trajectory

        Returns:
            List[Union[float, List[float]]]: Modified rewards with non-"stop" cases set to 0
        """
        for i, stop_reason in enumerate(stop_reasons):
            if stop_reason != "stop":
                if isinstance(rewards[i], list):
                    rewards[i] = [0.0] * len(rewards[i])
                else:
                    rewards[i] = 0.0
        return rewards

    # ----------------------------------------------------------------------------
    # Three methods of managing chat history and input ids in `agent_loop()`
    # ----------------------------------------------------------------------------
    def _update_agent_state_by_retokenizing_chat_history(
        self,
        agent_loop_state: AgentLoopState,
        turn_output: TurnOutput,
    ) -> AgentLoopState:
        """
        Update the chat history and input ids given a new model response and observation by retokenizing
        the entire chat history. Hence token-in-token-out is not followed.

        This method is used when `use_conversation_multi_turn=True` and `custom_chat_template` is set.
        It re-tokenizes the entire chat history every turn, which is useful for cases like removing
        Qwen3 thinking tokens in non-last-round assistant messages.

        Args:
            agent_loop_state: Current agent loop state containing chat history and input IDs
            turn_output: Turn output containing the model's response and new observations

        Returns:
            AgentLoopState: Updated agent loop state with retokenized chat history and input IDs.
                Note: loss_mask, response_end_idx, and rollout_logprobs are set to None as they
                are computed at the end with the custom chat template.
        """
        assert self.use_conversation_multi_turn and self.custom_chat_template

        agent_loop_state.chat_history = self._update_chat_history(
            agent_loop_state.chat_history, turn_output.output, turn_output.new_obs
        )

        # `loss_mask` is computed at the end with `custom_chat_template`
        agent_loop_state.loss_mask = None
        # untracked state
        agent_loop_state.response_end_idx = None
        # `logprobs` are not computed because retokenizing breaks token-in-token-out
        agent_loop_state.rollout_logprobs = None
        return agent_loop_state

    def _update_agent_loop_state_with_multiturn_chat_template(
        self,
        agent_loop_state: AgentLoopState,
        turn_output: TurnOutput,
    ) -> AgentLoopState:
        """
        Update the loss mask and input ids given a new model response and observation, following
        token-in-token-out.

        This function is used if `use_conversation_multi_turn` is True. It assumes that the input to the LLM is formatted as a list of messages, with observations
        stored in user messages.

        For example (using the Qwen 2.5 chat template), a trajectory for multi-turn generation would look like:
        <|im_start|>system
        ...
        <|im_end|>
        <|im_start|>user
                            question goes here
        <|im_end|>
        <|im_start|>assistant
                            turn 1 model response goes here
                            <think>... </think>
                            ...
        <|im_end|>
        <|im_start|>user
                            turn 1 env observation goes here
                            <observation>...</observation>
        <|im_end|>
        ...

        The chat template is applied without tokenization before and after the chat history is appended to
        in order to get new token ids in the chat template format (but without re-tokenizing the entire chat history every turn).

        Args:
            agent_loop_state: Current agent loop state containing chat history, input IDs, loss mask, etc.
            turn_output: Turn output containing the model's response, output IDs, logprobs, and observations

        Returns:
            AgentLoopState: Updated agent loop state with appended turn IDs, loss mask, and logprobs.
                For step-wise training, only response_end_idx is updated; loss_mask and rollout_logprobs
                are set to None as they are tracked per-step.
        """
        agent_loop_state.chat_history = self._update_chat_history(
            agent_loop_state.chat_history, turn_output.output, turn_output.new_obs
        )

        loss_mask_for_turn = turn_output.get_turn_loss_mask()
        rollout_logprobs_for_turn = turn_output.get_turn_rollout_logprobs()

        if self.generator_cfg.step_wise_trajectories:
            # cumulative input_ids is not tracked for step wise training
            agent_loop_state.response_end_idx = len(turn_output.output_ids) - 1
            # no running loss_mask or `rollout_logprobs` are tracked for step-wise training
            agent_loop_state.loss_mask = None
            agent_loop_state.rollout_logprobs = None
        else:
            # Directly append turn output
            turn_ids = turn_output.output_ids + turn_output.obs_ids
            agent_loop_state.response_end_idx = len(agent_loop_state.input_ids) + len(turn_output.output_ids) - 1
            agent_loop_state.input_ids += turn_ids
            agent_loop_state.loss_mask += loss_mask_for_turn
            if agent_loop_state.rollout_logprobs is not None and rollout_logprobs_for_turn is not None:
                agent_loop_state.rollout_logprobs += rollout_logprobs_for_turn

        return agent_loop_state

    def _update_agent_loop_state_with_singleturn_chat_template(
        self,
        agent_loop_state: AgentLoopState,
        turn_output: TurnOutput,
    ) -> AgentLoopState:
        """
        Update the loss mask and input ids given a new model response and observation, following
        token-in-token-out.

        This function is used if `use_conversation_multi_turn` is False. It assumes that the input to the LLM is a list of token ids
        and that the multi-turn conversation happens in a single assistant message.

        For example (using the Qwen 2.5 chat template), a trajectory for single-turn generation would look like:
        <|im_start|>system
        ...
        <|im_end|>
        <|im_start|>user
                            question goes here
        <|im_end|>
        <|im_start|>assistant
                            turn 1 model response goes here
                            <think>... </think>
                            ...

                            turn 1 env observation goes here
                            <observation>...</observation>

                            turn 2 model response goes here:
                            <think>... </think>
                            ...

        Args:
            agent_loop_state: Current agent loop state containing chat history, input IDs, loss mask, etc.
            turn_output: Turn output containing the model's response, output IDs, logprobs, and observations

        Returns:
            AgentLoopState: Updated agent loop state with appended turn IDs, loss mask, and logprobs.
                The EOS token is removed from response tokens (if present) since we are continuing
                the current assistant message. Observations are encoded directly without chat template formatting.
        """
        agent_loop_state.chat_history = self._update_chat_history(
            agent_loop_state.chat_history, turn_output.output, turn_output.new_obs
        )

        obs_ids_to_add = turn_output.obs_ids

        # Remove EOS token from response tokens since we are continuing the current assistant message
        new_resp_tokens = turn_output.output_ids.copy()
        if new_resp_tokens and new_resp_tokens[-1] == self.tokenizer.eos_token_id:
            new_resp_tokens = new_resp_tokens[:-1]

        turn_ids = new_resp_tokens + obs_ids_to_add
        loss_mask_for_turn = [1] * len(new_resp_tokens) + [0] * len(obs_ids_to_add)
        rollout_logprobs_for_turn = None
        if turn_output.output_logprobs is not None:
            # For response tokens, use actual logprobs
            # for obs tokens, use dummy values
            rollout_logprobs_for_turn = turn_output.output_logprobs[: len(new_resp_tokens)] + [0.0] * len(
                obs_ids_to_add
            )

        # Directly append turn output
        agent_loop_state.response_end_idx = len(agent_loop_state.input_ids) + len(new_resp_tokens) - 1
        agent_loop_state.input_ids += turn_ids
        agent_loop_state.loss_mask += loss_mask_for_turn
        if agent_loop_state.rollout_logprobs is not None and rollout_logprobs_for_turn is not None:
            agent_loop_state.rollout_logprobs += rollout_logprobs_for_turn

        return agent_loop_state
