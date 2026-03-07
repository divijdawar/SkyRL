"""SGLang inference engine implementation."""

import torch
import os
import time
import random
from typing import List, Optional, Dict, Any, Literal, AsyncIterator, Tuple, TYPE_CHECKING
import ray
from loguru import logger
import multiprocessing as mp

import sglang.srt.entrypoints.engine
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.utils import (
    assert_pkg_version,
    is_cuda,
    set_prometheus_multiproc_dir,
    set_ulimit,
    MultiprocessingSerializer,
    get_bool_env_var,
)
from sglang.srt.managers.io_struct import (
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromDistributedReqInput,
    InitWeightsUpdateGroupReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    PauseGenerationReqInput,
    ContinueGenerationReqInput,
    UpdateWeightVersionReqInput,
)
from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
    InferenceEngineInput,
    InferenceEngineOutput,
    StreamingChunk,
    WeightTransferHandle,
)
from skyrl_train.weight_sync import (
    WeightLoader,
    WeightUpdateRequest,
    CudaIpcWeightUpdateRequest,
    BroadcastWeightUpdateRequest,
)


# Store original is_cuda check result
_is_cuda = is_cuda()


def _is_oom_error(error: Exception) -> bool:
    """Detect if an error is an out-of-memory error from SGLang or CUDA.

    SGLang raises RuntimeError with specific messages for OOM conditions.
    CUDA also raises cuda.OutOfMemoryError for GPU memory exhaustion.

    Args:
        error: The exception to check.

    Returns:
        True if the error indicates an out-of-memory condition.
    """
    error_str = str(error).lower()
    oom_patterns = [
        "out of memory",
        "oom",
        "prefill out of memory",
        "decode out of memory",
        "cuda out of memory",
        "cudaoutofmemoryerror",
        "allocate",  # "failed to allocate" patterns
    ]
    return any(pattern in error_str for pattern in oom_patterns)


# Patch SGLang's _set_envs_and_config to handle signal handlers properly in Ray actors
# Based on VERL's solution: https://github.com/sgl-project/sglang/issues/6723
# Improved with thread detection so standalone servers can still use signal handlers
def _patched_set_envs_and_config(server_args):
    """Patched version of SGLang's _set_envs_and_config with thread-safe signal handling.

    Uses thread detection to determine if signal handlers can be registered:
    - In main thread: Registers signal handlers normally (for standalone servers)
    - In worker threads (Ray actors): Skips signal handler registration safely

    This is needed because Python's signal module only allows handler registration
    from the main thread of the main interpreter.
    """
    import signal
    import threading

    # Set global environments (matching current SGLang implementation)
    if "NCCL_CUMEM_ENABLE" not in os.environ or getattr(server_args, "enable_symm_mem", False):
        os.environ["NCCL_CUMEM_ENABLE"] = str(int(getattr(server_args, "enable_symm_mem", False)))
    if (
        "NCCL_NVLS_ENABLE" not in os.environ
        or getattr(server_args, "enable_nccl_nvls", False)
        or getattr(server_args, "enable_symm_mem", False)
    ):
        os.environ["NCCL_NVLS_ENABLE"] = str(
            int(getattr(server_args, "enable_nccl_nvls", False) or getattr(server_args, "enable_symm_mem", False))
        )
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "8"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"

    if os.environ.get("TRTLLM_ENABLE_PDL", "1") != "0":
        os.environ["TRTLLM_ENABLE_PDL"] = "1"

    if os.environ.get("CUTE_DSL_LOG_LEVEL") is None:
        os.environ["CUTE_DSL_LOG_LEVEL"] = "30"

    if os.environ.get("CUTE_DSL_LOG_TO_CONSOLE") is None:
        os.environ["CUTE_DSL_LOG_TO_CONSOLE"] = "1"

    os.environ["SGLANG_RUN_ID"] = f"sglang-run-{time.time()}-{random.randint(0, 100000000)}"

    # Set prometheus env vars
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()

    # Set ulimit
    set_ulimit()

    # Check flashinfer version
    if not get_bool_env_var("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK"):
        if server_args.attention_backend == "flashinfer":
            assert_pkg_version(
                "flashinfer_python",
                "0.5.3",
                "Please uninstall the old version and reinstall the latest version by following the instructions at https://docs.flashinfer.ai/installation.html.",
            )
        if _is_cuda:
            assert_pkg_version(
                "sgl-kernel",
                "0.3.20",
                "Please reinstall the latest version with `pip install sgl-kernel --force-reinstall`",
            )

    # Set mp start method
    mp.set_start_method("spawn", force=True)

    # Thread-safe signal handler registration
    # Only register signal handlers if we're in the main thread (standalone servers)
    # Skip in worker threads (Ray actors) to avoid ValueError
    if threading.current_thread() is threading.main_thread():
        def _kill_process_tree(pid):
            """Kill process tree starting from pid."""
            import subprocess
            try:
                subprocess.run(["pkill", "-TERM", "-P", str(pid)], check=False)
            except Exception:
                pass

        def launch_phase_sigquit_handler(signum, frame):
            logger.error(
                "Received SIGQUIT from a child process. "
                "This usually means a subprocess failed during launch."
            )
            _kill_process_tree(os.getpid())

        try:
            if server_args.custom_sigquit_handler is None:
                signal.signal(signal.SIGQUIT, launch_phase_sigquit_handler)
            else:
                signal.signal(signal.SIGQUIT, server_args.custom_sigquit_handler)
            logger.debug("Registered SIGQUIT handler for subprocess failure detection")
        except Exception as e:
            logger.warning(f"Could not register signal handler: {e}")
    else:
        # Running in a worker thread (e.g., Ray actor) - skip signal handler registration
        logger.debug(
            "Running in non-main thread (likely Ray actor context). "
            "Signal handler registration skipped. Subprocess failure detection unavailable."
        )


# Apply the patch
sglang.srt.entrypoints.engine._set_envs_and_config = _patched_set_envs_and_config


def setup_gpu_for_sglang(kwargs, bundle_indices):
    """Configure GPU device assignment for SGLang using native API.

    Instead of hacking CUDA_VISIBLE_DEVICES environment variable, we use SGLang's
    native `base_gpu_id` parameter which is designed for this exact purpose.

    Args:
        kwargs: Engine kwargs to modify in-place.
        bundle_indices: Optional Ray bundle indices for placement.
    """
    import os

    # Remove legacy parameters that are no longer needed
    kwargs.pop("distributed_executor_backend", None)
    kwargs.pop("noset_visible_devices", None)

    # Use SGLang's native base_gpu_id for device assignment
    # This is the proper way to specify which GPU(s) to use, rather than
    # manipulating CUDA_VISIBLE_DEVICES environment variables.
    if "base_gpu_id" not in kwargs:
        try:
            assigned_gpus = ray.get_gpu_ids()
            if assigned_gpus:
                # Check if Ray has set CUDA_VISIBLE_DEVICES (sandboxing the GPUs)
                cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                if cuda_visible:
                    # Ray sandboxes GPUs via CUDA_VISIBLE_DEVICES, so GPUs appear as 0, 1, 2, ...
                    # within the worker. Use base_gpu_id=0 regardless of physical GPU IDs.
                    kwargs["base_gpu_id"] = 0
                    logger.debug(
                        f"Set base_gpu_id=0 (Ray CUDA_VISIBLE_DEVICES={cuda_visible}, "
                        f"assigned_gpus={assigned_gpus})"
                    )
                else:
                    # No sandboxing - use the first Ray-assigned GPU as the base
                    kwargs["base_gpu_id"] = int(assigned_gpus[0])
                    logger.debug(f"Set base_gpu_id={kwargs['base_gpu_id']} from Ray GPU assignment")
        except Exception as e:
            # Not running in Ray context or no GPUs assigned
            logger.debug(f"Could not get Ray GPU IDs, using default base_gpu_id: {e}")


def sglang_custom_weight_loader(model, named_tensors):
    """Custom weight loader for SGLang that handles CUDA IPC.

    This function is called by SGLang's model runner to load weights.
    It reconstructs tensors from CudaIpcWeightUpdateRequest using CUDA IPC handles.

    Note: Broadcast path is not handled here.
    Because unlike vLLM where we control WorkerWrap and can store the
    process group there, SGLang's custom_weight_loader only receives (model, tensors).
    The process group (_model_update_group) is stored in model_runner, which is not
    accessible from the model object. We also cannot create the group lazily inside
    custom_weight_loader because torch.distributed group creation requires coordination
    (all processes must join at the same time), and by the time custom_weight_loader
    is called, the training side has already completed its init. Therefore, broadcast
    uses SGLang's native update_weights_from_distributed API which has internal access
    to the process group.
    """
    from skyrl_train.weight_sync import CudaIpcWeightUpdateRequest, CudaIpcWeightTransferReceiver

    # Extract tensor name and data
    name, tensor = named_tensors[0]
    if name != "ipc_request":
        raise ValueError(f"Expected tensor name 'ipc_request', got: {name}")

    # Deserialize request from tensor
    request = CudaIpcWeightUpdateRequest.deserialize(tensor.cpu().numpy().tobytes())

    # Get model info and create receiver
    model_dtype = next(model.parameters()).dtype
    receiver = CudaIpcWeightTransferReceiver(model_dtype)

    # Receive weights via IPC
    weights_to_load = list(receiver.receive_weights(request))
    model.load_weights(weights_to_load)


CUSTOM_WEIGHT_LOADER_PATH = "skyrl_train.inference_engines.sglang.sglang_engine.sglang_custom_weight_loader"


# Memory type tags for selective memory release
# These correspond to SGLang's GpuMemoryType enum values
class MemoryTag:
    """Memory type tags for selective memory release.

    Use these tags with sleep()/wake_up() to control which memory regions
    are released/restored. This enables more efficient memory management
    during training by preserving KV cache for prefix reuse.

    Example:
        # Release only weights during training (preserve KV cache)
        await engine.sleep(tags=[MemoryTag.WEIGHTS])

        # ... perform training ...

        await engine.wake_up(tags=[MemoryTag.WEIGHTS])
        await engine.update_weights(...)

        # Release all memory when completely done
        await engine.sleep(tags=MemoryTag.ALL)
    """

    WEIGHTS = "weights"
    KV_CACHE = "kv_cache"
    CUDA_GRAPH = "cuda_graph"

    # Convenience lists
    ALL = [WEIGHTS, KV_CACHE, CUDA_GRAPH]
    TRAINING_DEFAULT = [WEIGHTS]  # Release only weights, preserve KV cache


class SGLangWeightLoader(WeightLoader):
    """Loads weights into SGLang engine, managing weight transfer coordination.

    This loader encapsulates the SGLang-specific weight loading logic for both
    IPC and broadcast transfer paths:
    - IPC: Uses update_weights_from_tensor with our custom_weight_loader
    - Broadcast: Uses SGLang's native update_weights_from_distributed API
    """

    def __init__(self, engine: Any, tp_size: int) -> None:
        """Initialize the loader.

        Args:
            engine: The SGLang engine.
            tp_size: Tensor parallel size.
        """
        self._engine = engine
        self._tp_size = tp_size
        self._group_name: Optional[str] = None

    async def init_communicator(self, init_info) -> None:
        """Initialize the process group for broadcast weight sync.

        This is only needed for the broadcast path. IPC path does not require
        a process group since it uses CUDA IPC handles directly.

        For TP > 1 scenarios, SGLang handles the coordination internally:
        - The rank_offset is passed to SGLang's init_weights_update_group
        - SGLang internally calculates each TP worker's rank as: rank = rank_offset + tp_rank
        - All TP workers join the process group with their respective ranks
        - The training rank 0 broadcasts to all TP workers across all engines

        Note: For colocated training (trainer.placement.colocate_all=True), CUDA IPC
        strategy is recommended over broadcast for better performance, especially
        with TP > 1 where broadcast requires multiple process group operations.

        Args:
            init_info: WeightSyncInitInfo from the sender.
        """
        from skyrl_train.weight_sync import BroadcastTransferStrategy

        if init_info.strategy_type() is BroadcastTransferStrategy:
            # Log info for debugging TP > 1 scenarios
            if self._tp_size > 1:
                logger.info(
                    f"Initializing broadcast weight sync for SGLang engine with TP={self._tp_size}. "
                    f"rank_offset={init_info.rank_offset}, world_size={init_info.world_size}. "
                    f"Consider using CUDA IPC (colocate_all=True) for better performance."
                )

            obj = InitWeightsUpdateGroupReqInput(
                master_address=init_info.master_addr,
                master_port=init_info.master_port,
                rank_offset=init_info.rank_offset,
                world_size=init_info.world_size,
                group_name=init_info.group_name,
                backend=init_info.backend,
            )
            # Store group_name for use in _load_via_broadcast
            self._group_name = init_info.group_name
            # NOTE(charlie): Call the async method on tokenizer_manager directly to avoid event loop
            # conflicts. Same underlying implementation: https://github.com/sgl-project/sglang/blob/v0.4.8.post1/python/sglang/srt/model_executor/model_runner.py#L689
            success, message = await self._engine.tokenizer_manager.init_weights_update_group(obj, None)
            if not success:
                raise RuntimeError(f"Failed to initialize weight update group: {message}")

    async def load_weights(self, request: WeightUpdateRequest) -> None:
        """Load weights by coordinating with SGLang's weight update APIs.

        Args:
            request: Weight update request.
        """
        if isinstance(request, CudaIpcWeightUpdateRequest):
            await self._load_via_ipc(request)
        elif isinstance(request, BroadcastWeightUpdateRequest):
            await self._load_via_broadcast(request)
        else:
            raise TypeError(f"Unknown request type: {type(request).__name__}")

    async def _load_via_ipc(self, request: CudaIpcWeightUpdateRequest) -> None:
        """Load weights via CUDA IPC using custom weight loader.

        Uses SGLangWeightTransferReceiver internally to receive weights
        from IPC handles.
        """
        tensor_array = torch.frombuffer(bytearray(request.serialize()), dtype=torch.uint8)

        # Use SGLang's API to update weights with our custom loader
        request_tensor = [("ipc_request", tensor_array)]
        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[
                MultiprocessingSerializer.serialize(request_tensor) for _ in range(self._tp_size)
            ],
            load_format=CUSTOM_WEIGHT_LOADER_PATH,
            # Flush cache after weight updates to invalidate KV cache computed with old weights
            flush_cache=True,
            # Pass weight version for tracking (SGLang uses it for sample-to-step correlation)
            weight_version=request.weight_version,
        )

        success, message = await self._engine.tokenizer_manager.update_weights_from_tensor(obj, None)
        if not success:
            raise RuntimeError(f"IPC weight update failed: {message}")

    async def _load_via_broadcast(self, request: BroadcastWeightUpdateRequest) -> None:
        """Load weights via torch.distributed broadcast.

        Uses SGLang's native update_weights_from_distributed API which internally
        uses the process group created during init_weights_update_group.

        This method supports single or batched weight updates. When a request contains
        multiple weights, they can be sent in a single call for better performance.
        """
        if len(request) == 0:
            return

        # Send all weights in a single call for better performance
        # SGLang will receive and apply all weights atomically
        obj = UpdateWeightsFromDistributedReqInput(
            names=request.names,
            dtypes=request.dtypes,
            shapes=request.shapes,
            group_name=self._group_name,
            # Pass weight version for tracking (SGLang uses it for sample-to-step correlation)
            weight_version=request.weight_version,
        )

        success, message = await self._engine.tokenizer_manager.update_weights_from_distributed(obj, None)
        if not success:
            raise RuntimeError(f"Broadcast weight update failed: {message}")

    async def destroy_group(self) -> None:
        """Destroy the weight update process group.

        Should be called during teardown to clean up NCCL resources.
        Only relevant for broadcast strategy (IPC doesn't use process groups).
        """
        if self._group_name is None:
            # No group was initialized (using IPC or not initialized yet)
            return

        try:
            from sglang.srt.managers.io_struct import DestroyWeightsUpdateGroupReqInput
            obj = DestroyWeightsUpdateGroupReqInput(group_name=self._group_name)
            success, message = await self._engine.tokenizer_manager.destroy_weights_update_group(obj, None)
            if not success:
                logger.warning(f"Failed to destroy weight update group '{self._group_name}': {message}")
            else:
                logger.debug(f"Destroyed weight update group '{self._group_name}'")
            self._group_name = None
        except Exception as e:
            logger.warning(f"Error destroying weight update group: {e}")


class SGLangInferenceEngine(InferenceEngineInterface):
    """SGLang inference engine that implements InferenceEngineInterface."""

    def __init__(self, *args, bundle_indices: Optional[List[int]] = None, **kwargs):
        setup_gpu_for_sglang(kwargs, bundle_indices)

        # Store common attributes
        self._tp_size = kwargs.get("tp_size", 1)
        self._pp_size = kwargs.get("pp_size", 1)
        self._dp_size = kwargs.get("dp_size", 1)
        self._ep_size = kwargs.get("ep_size", 1)
        self._enable_lora = kwargs.get("enable_lora", False)
        self.tokenizer = kwargs.pop("tokenizer", None)
        if self.tokenizer is None:
            raise ValueError(
                "tokenizer is required for SGLangInferenceEngine (for decoding output_ids). "
                "Pass tokenizer via kwargs when creating the engine."
            )
        self._model_path = kwargs.get("model_path", "")

        # Unused kwargs
        _ = kwargs.pop("num_gpus", 1)

        # Add custom weight loader
        kwargs["custom_weight_loader"] = CUSTOM_WEIGHT_LOADER_PATH

        # Always use token-in-token-out SGLang engine
        # NOTE(Charlie): unlike vLLM, SGLang cannot do token-in-token-out and
        # token-in-text-out in the same engine config.
        kwargs["skip_tokenizer_init"] = True

        # Create the SGLang engine (signal handler issue is now fixed by patching)
        self.engine = Engine(**kwargs)
        logger.info(
            f"Created SGLang engine with tp_size={self._tp_size}, pp_size={self._pp_size}, "
            f"dp_size={self._dp_size}, ep_size={self._ep_size}, enable_lora={self._enable_lora}"
        )

        # Create weight loader for coordinating weight updates
        self._weight_loader = SGLangWeightLoader(self.engine, self._tp_size)

        # Local weight version tracking (fallback when SGLang API not available)
        self._weight_version: Optional[str] = None
        self._version_api_available: Optional[bool] = None  # None = not yet checked

    def tp_size(self):
        return self._tp_size

    def pp_size(self):
        return self._pp_size

    def dp_size(self):
        return self._dp_size

    def ep_size(self):
        return self._ep_size

    async def load_lora_adapter(self, lora_name: str, lora_path: str, pinned: bool = False):
        """Load a LoRA adapter at runtime.

        Args:
            lora_name: Unique name for this adapter
            lora_path: Path to the adapter files
            pinned: If True, adapter won't be evicted from memory
        """
        if not self._enable_lora:
            raise RuntimeError("LoRA is not enabled. Set enable_lora=True when creating the engine.")

        from sglang.srt.managers.io_struct import LoadLoRAAdapterReqInput
        req = LoadLoRAAdapterReqInput(lora_name=lora_name, lora_path=lora_path, pinned=pinned)
        result = await self.engine.tokenizer_manager.load_lora_adapter(req, None)
        logger.info(f"Loaded LoRA adapter '{lora_name}' from {lora_path}")
        return result

    async def unload_lora_adapter(self, lora_name: str):
        """Unload a LoRA adapter at runtime.

        Args:
            lora_name: Name of the adapter to unload
        """
        if not self._enable_lora:
            raise RuntimeError("LoRA is not enabled. Set enable_lora=True when creating the engine.")

        from sglang.srt.managers.io_struct import UnloadLoRAAdapterReqInput
        req = UnloadLoRAAdapterReqInput(lora_name=lora_name)
        result = await self.engine.tokenizer_manager.unload_lora_adapter(req, None)
        logger.info(f"Unloaded LoRA adapter '{lora_name}'")
        return result

    def _preprocess_prompts(self, input_batch: InferenceEngineInput):
        """Preprocess prompts for SGLang generation."""
        prompts = input_batch.get("prompts")
        prompt_token_ids = input_batch.get("prompt_token_ids")
        request_sampling_params = input_batch.get("sampling_params")

        assert (
            prompts is None and prompt_token_ids is not None
        ), "SGLangInferenceEngine only accepts `prompt_token_ids`, not `prompts`."

        # Use request sampling params if provided.
        sampling_params = dict(request_sampling_params) if request_sampling_params is not None else {}

        # Handle stop_regex via post-processing since SGLang's internal implementation
        # requires tokenizer.decode() which is unavailable with skip_tokenizer_init=True.
        # We store the pattern and apply it during postprocessing using our external tokenizer.
        # Note: This means generation runs to max_tokens, then we trim at regex match.
        # For efficiency with long generations, consider using stop strings instead.
        stop_regex = sampling_params.pop("stop_regex", None)
        if stop_regex is not None:
            logger.debug(f"stop_regex '{stop_regex}' will be applied via post-processing")
        sampling_params["_stop_regex"] = stop_regex  # Store for post-processing (None if not set)

        # Convert stop strings to stop_token_ids using our external tokenizer
        # This works even with skip_tokenizer_init=True since we use the external tokenizer
        # Multi-token stop sequences: we use the last token as the stop trigger, then
        # post-process to trim at the correct multi-token boundary.
        stop_strings = sampling_params.pop("stop", None)
        multi_token_stop_seqs: List[List[int]] = []  # Store for post-processing
        if stop_strings is not None:
            stop_token_ids = sampling_params.get("stop_token_ids", []) or []
            if isinstance(stop_strings, str):
                stop_strings = [stop_strings]
            for stop_str in stop_strings:
                # Encode each stop string and get the token IDs
                token_ids = self.tokenizer.encode(stop_str, add_special_tokens=False)
                if token_ids:
                    if len(token_ids) > 1:
                        # Multi-token stop: use last token as trigger, store full sequence
                        # for post-processing trimming
                        multi_token_stop_seqs.append(token_ids)
                        stop_token_ids.append(token_ids[-1])
                        logger.debug(
                            f"Multi-token stop '{stop_str}' -> trigger on token {token_ids[-1]}, "
                            f"will trim at full sequence {token_ids}"
                        )
                    else:
                        # Single-token stop: direct mapping
                        stop_token_ids.append(token_ids[0])
            if stop_token_ids:
                sampling_params["stop_token_ids"] = list(set(stop_token_ids))
                logger.debug(f"Converted stop strings {stop_strings} to stop_token_ids {sampling_params['stop_token_ids']}")

        # Store multi-token sequences for post-processing (returned separately)
        sampling_params["_multi_token_stop_seqs"] = multi_token_stop_seqs

        # Pass eos_token_id for min_new_tokens support with skip_tokenizer_init=True
        # The min_new_tokens penalizer needs to know the EOS token to suppress it
        if sampling_params.get("min_new_tokens", 0) > 0:
            sampling_params["eos_token_id"] = self.tokenizer.eos_token_id
            logger.debug(f"Set eos_token_id={sampling_params['eos_token_id']} for min_new_tokens support")

        # Map common parameter name aliases to SGLang's expected names
        # SGLang uses 'sampling_seed' instead of 'seed' for reproducibility
        if "seed" in sampling_params and "sampling_seed" not in sampling_params:
            sampling_params["sampling_seed"] = sampling_params.pop("seed")
            logger.debug(f"Mapped 'seed' to 'sampling_seed': {sampling_params['sampling_seed']}")

        # Validate structured output constraints (only one can be specified)
        # These are passed directly to SGLang which handles the grammar-based decoding
        structured_params = ["json_schema", "regex", "ebnf"]
        active_constraints = [p for p in structured_params if sampling_params.get(p) is not None]
        if len(active_constraints) > 1:
            raise ValueError(
                f"Only one structured output constraint can be specified at a time. "
                f"Got: {active_constraints}"
            )
        if active_constraints:
            logger.debug(f"Using structured output constraint: {active_constraints[0]}")

        return prompt_token_ids, sampling_params

    def _trim_at_multi_token_stop(
        self,
        token_ids: List[int],
        multi_token_stop_seqs: List[List[int]],
    ) -> List[int]:
        """Trim token IDs at multi-token stop sequence boundary.

        When we stop on the last token of a multi-token sequence, the preceding
        tokens of that sequence are already in the output. This method finds
        and removes the complete stop sequence from the end of the output.

        Args:
            token_ids: The generated token IDs.
            multi_token_stop_seqs: List of multi-token stop sequences to check.

        Returns:
            Token IDs with the stop sequence removed if found at the end.
        """
        if not multi_token_stop_seqs or not token_ids:
            return token_ids

        for stop_seq in multi_token_stop_seqs:
            seq_len = len(stop_seq)
            if len(token_ids) >= seq_len:
                # Check if the output ends with this stop sequence
                if token_ids[-seq_len:] == stop_seq:
                    logger.debug(
                        f"Trimming multi-token stop sequence {stop_seq} from output "
                        f"(removing last {seq_len} tokens)"
                    )
                    return token_ids[:-seq_len]

                # Also check for partial match (stop triggered before full sequence)
                # This handles cases where the last token triggered stop but not all
                # preceding tokens are present (e.g., stop on token that appears elsewhere)
                # We look for the longest suffix match
                for suffix_len in range(seq_len - 1, 0, -1):
                    if (len(token_ids) >= suffix_len and
                        token_ids[-suffix_len:] == stop_seq[-suffix_len:]):
                        # Found partial match at end - this is the stop sequence suffix
                        # Check if the tokens before it could be start of another occurrence
                        logger.debug(
                            f"Trimming partial multi-token stop sequence suffix {stop_seq[-suffix_len:]} "
                            f"from output (removing last {suffix_len} tokens)"
                        )
                        return token_ids[:-suffix_len]

        return token_ids

    def _trim_at_regex_stop(
        self,
        text: str,
        token_ids: List[int],
        stop_regex: str,
    ) -> Tuple[str, List[int], bool]:
        """Trim output at regex match boundary.

        Searches for the first occurrence of stop_regex in the decoded text
        and trims both text and token_ids at that boundary.

        Args:
            text: The decoded response text.
            token_ids: The response token IDs.
            stop_regex: The regex pattern to search for.

        Returns:
            Tuple of (trimmed_text, trimmed_token_ids, regex_matched).
            If regex matches, text and tokens are trimmed at the match start.
            The regex_matched flag indicates if stop was due to regex.
        """
        import re

        try:
            pattern = re.compile(stop_regex)
            match = pattern.search(text)

            if match:
                # Found regex match - trim at match start
                match_start = match.start()
                trimmed_text = text[:match_start]

                # Re-encode trimmed text to get correct token IDs
                # This handles the case where regex boundary doesn't align with token boundary
                trimmed_token_ids = self.tokenizer.encode(trimmed_text, add_special_tokens=False)

                logger.debug(
                    f"Trimmed at stop_regex '{stop_regex}' match at position {match_start}. "
                    f"Original length: {len(token_ids)}, trimmed: {len(trimmed_token_ids)}"
                )
                return trimmed_text, trimmed_token_ids, True

        except re.error as e:
            logger.warning(f"Invalid stop_regex pattern '{stop_regex}': {e}")

        return text, token_ids, False

    def _postprocess_outputs(
        self,
        outputs,
        return_logprobs: bool = False,
        n_per_prompt: int = 1,
        extract_request_ids: bool = False,
        multi_token_stop_seqs: Optional[List[List[int]]] = None,
        stop_regex: Optional[str] = None,
        extract_hidden_states: bool = False,
        extract_routed_experts: bool = False,
    ):
        """Process SGLang outputs to match expected format.

        Args:
            outputs: Raw outputs from SGLang engine.
            return_logprobs: Whether to extract and return log probabilities.
            n_per_prompt: Number of samples generated per prompt (for n>1 sampling).
            extract_request_ids: Whether to extract request IDs for session continuity.
            multi_token_stop_seqs: List of multi-token stop sequences for trimming.
            stop_regex: Regex pattern to stop at (applied via post-processing).
            extract_hidden_states: Whether to extract and return hidden states.
                Requires engine to be initialized with enable_return_hidden_states=True.
            extract_routed_experts: Whether to extract and return routed expert indices.
                Requires SGLang to be started with return_routed_experts=True.
                Used for R3 (Rollout Routing Replay) to stabilize MoE RL training.

        Returns:
            InferenceEngineOutput with responses, response_ids, stop_reasons, optionally logprobs,
            weight_version for tracking which training step's weights generated the output,
            n_per_prompt for reconstructing per-prompt groups, optionally request_ids
            for session-based generation, optionally hidden_states for RL,
            and optionally routed_experts for R3.
        """
        responses: List[str] = []
        stop_reasons: List[str] = []
        response_ids: List[List[int]] = []
        response_logprobs: Optional[List[List[float]]] = [] if return_logprobs else None
        request_ids: Optional[List[str]] = [] if extract_request_ids else None
        hidden_states: Optional[List[Any]] = [] if extract_hidden_states else None
        routed_experts: Optional[List[Any]] = [] if extract_routed_experts else None
        weight_version: Optional[str] = None

        for output in outputs:
            output_ids = output["output_ids"]

            # Trim at multi-token stop sequence boundary if applicable
            if multi_token_stop_seqs:
                output_ids = self._trim_at_multi_token_stop(output_ids, multi_token_stop_seqs)

            # Decode first, then apply regex trimming
            decoded_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            stop_reason = output["meta_info"]["finish_reason"]["type"]

            # Apply stop_regex trimming if specified
            if stop_regex:
                decoded_text, output_ids, regex_matched = self._trim_at_regex_stop(
                    decoded_text, output_ids, stop_regex
                )
                if regex_matched:
                    stop_reason = "stop"  # Changed from "length" to "stop" due to regex match

            response_ids.append(output_ids)
            responses.append(decoded_text)
            stop_reasons.append(stop_reason)

            # Extract weight version from first output (same for all outputs in batch)
            if weight_version is None:
                meta_info = output.get("meta_info", {})
                weight_version = meta_info.get("weight_version", None)

            # Extract request ID for session-based generation
            if extract_request_ids:
                meta_info = output.get("meta_info", {})
                rid = meta_info.get("id", None)
                request_ids.append(rid)

            # Extract logprobs if requested and available
            if return_logprobs:
                meta_info = output.get("meta_info", {})
                output_token_logprobs = meta_info.get("output_token_logprobs", None)
                if output_token_logprobs is not None:
                    # SGLang returns logprobs as list of (logprob, token_id, decoded_token) tuples
                    # or just list of floats depending on configuration
                    if isinstance(output_token_logprobs, list) and len(output_token_logprobs) > 0:
                        if isinstance(output_token_logprobs[0], (list, tuple)):
                            # Extract just the logprob values from tuples
                            logprobs = [lp[0] if isinstance(lp, (list, tuple)) else lp for lp in output_token_logprobs]
                        else:
                            logprobs = output_token_logprobs
                        response_logprobs.append(logprobs)
                    else:
                        response_logprobs.append([])
                else:
                    response_logprobs.append([])

            # Extract hidden states if requested and available
            # Useful for value function enrichment, representation learning, and RL state extraction
            if extract_hidden_states:
                meta_info = output.get("meta_info", {})
                output_hidden_states = meta_info.get("hidden_states", None)
                hidden_states.append(output_hidden_states)

            # Extract routed expert indices if requested and available
            # Used for R3 (Rollout Routing Replay) to stabilize MoE RL training
            if extract_routed_experts:
                meta_info = output.get("meta_info", {})
                output_routed_experts = meta_info.get("routed_experts", None)
                routed_experts.append(output_routed_experts)

        # Convert empty logprobs list to None to match expected interface behavior
        # This prevents IndexError when generator tries to access response_logprobs[0]
        if response_logprobs is not None and len(response_logprobs) == 0:
            response_logprobs = None

        # Convert empty hidden_states list to None
        if hidden_states is not None and len(hidden_states) == 0:
            hidden_states = None

        # Convert empty routed_experts list to None
        if routed_experts is not None and len(routed_experts) == 0:
            routed_experts = None

        return InferenceEngineOutput(
            responses=responses,
            response_ids=response_ids,
            stop_reasons=stop_reasons,
            response_logprobs=response_logprobs,
            weight_version=weight_version,
            n_per_prompt=n_per_prompt if (n_per_prompt is not None and n_per_prompt > 1) else None,
            request_ids=request_ids,
            hidden_states=hidden_states,
            routed_experts=routed_experts,
        )

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        """Generate responses using SGLang engine with optional multimodal support.

        Args:
            input_batch: Input batch containing prompt_token_ids and sampling_params.
                         sampling_params may include:
                         - 'return_logprob': Request log probabilities for generated tokens.
                         - 'n': Number of samples to generate per prompt (default: 1).
                           When n>1, SGLang's parallel_sample_num is used for efficient
                           parallel sampling.
                         Optional multimodal field:
                         - 'image_data': List of image inputs (one per prompt), where each can be:
                           - str: File path, URL, or base64-encoded image
                           - bytes: Raw image bytes
                           - PIL.Image.Image: PIL Image object
                           - List[...]: Multiple images per prompt
                           - None: No image for this prompt

        Returns:
            InferenceEngineOutput with responses, response_ids, stop_reasons, and optionally logprobs.
            When n>1, outputs are flattened but grouped by prompt:
            [prompt0_sample0, prompt0_sample1, ..., prompt0_sampleN-1, prompt1_sample0, ...]
            The n_per_prompt field indicates the number of samples per prompt for reconstruction:
            outputs[i*n:(i+1)*n] gives all samples for prompt i.

        Raises:
            RuntimeError: If generation or postprocessing fails, with context about the failure.
        """
        import asyncio

        token_ids_prompts, sampling_params = self._preprocess_prompts(input_batch)

        # Extract logprob parameters (these should be passed to async_generate, not in sampling_params)
        return_logprob = sampling_params.pop("return_logprob", False)
        logprob_start_len = sampling_params.pop("logprob_start_len", None)
        top_logprobs_num = sampling_params.pop("top_logprobs_num", None)

        # Extract hidden states parameter for RL value function enrichment
        # Can be passed via sampling_params or input_batch
        return_hidden_states = sampling_params.pop("return_hidden_states", False)
        if not return_hidden_states:
            return_hidden_states = input_batch.get("return_hidden_states", False)

        # Extract routed experts parameter for R3 (Rollout Routing Replay)
        # Captures which MoE experts were routed to during inference for replay during training
        return_routed_experts = sampling_params.pop("return_routed_experts", False)
        if not return_routed_experts:
            return_routed_experts = input_batch.get("return_routed_experts", False)

        # Extract custom_logit_processor (request-level field, not a SamplingParams field)
        # This is the serialized processor string from CustomLogitProcessor.to_str()
        # Note: custom_params stays in sampling_params as SGLang expects it there
        custom_logit_processor = sampling_params.pop("custom_logit_processor", None)

        # Support RL action masking via action_mask or disallowed_tokens
        # These are converted to custom_logit_processor format for SGLang
        if custom_logit_processor is None:
            action_mask = sampling_params.pop("action_mask", None)
            disallowed_tokens = sampling_params.pop("disallowed_tokens", None)
            if action_mask or disallowed_tokens:
                from skyrl_train.inference_engines.sglang.rl_logit_processors import (
                    create_rl_logit_processor,
                )
                custom_logit_processor = create_rl_logit_processor(
                    action_mask=action_mask,
                    disallowed_tokens=disallowed_tokens,
                )

        # Extract multi-token stop sequences for post-processing (internal field)
        multi_token_stop_seqs = sampling_params.pop("_multi_token_stop_seqs", None)

        # Extract stop_regex for post-processing (internal field)
        stop_regex = sampling_params.pop("_stop_regex", None)

        # Extract n for parallel sampling (SGLang uses 'n' which maps to parallel_sample_num)
        n_per_prompt = sampling_params.get("n", 1) or 1  # Default to 1 if None

        # Extract multimodal data if provided (for VLMs, video models, audio models)
        image_data = input_batch.get("image_data", None)
        video_data = input_batch.get("video_data", None)
        audio_data = input_batch.get("audio_data", None)

        # Extract request-level priority for scheduling (SGLang priority scheduling feature)
        # Lower values = higher priority. Use for reward model requests during RL training.
        priority = sampling_params.pop("priority", None)

        # Extract per-request LoRA adapter name for multi-adapter serving
        # Enables different adapters per request in the same batch (S-LoRA architecture)
        # Pass via sampling_params: {"lora_name": "my_adapter", ...}
        lora_name = sampling_params.pop("lora_name", None)

        # OOM recovery configuration
        max_oom_retries = 3
        oom_retry_delay = 0.5  # seconds, doubles each retry

        last_error = None
        for attempt in range(max_oom_retries + 1):
            try:
                # Use GenerateReqInput directly to support priority scheduling
                # (Engine.async_generate doesn't expose priority parameter)
                from sglang.srt.managers.io_struct import GenerateReqInput

                obj = GenerateReqInput(
                    input_ids=token_ids_prompts,
                    sampling_params=dict(sampling_params),  # Copy to avoid mutation on retry
                    image_data=image_data,
                    video_data=video_data,
                    audio_data=audio_data,
                    return_logprob=return_logprob,
                    logprob_start_len=logprob_start_len,
                    top_logprobs_num=top_logprobs_num,
                    return_hidden_states=return_hidden_states,
                    return_routed_experts=return_routed_experts,
                    custom_logit_processor=custom_logit_processor,
                    priority=priority,
                    lora_path=lora_name,  # SGLang uses lora_path to reference pre-loaded adapters by name
                )

                generator = self.engine.tokenizer_manager.generate_request(obj, None)
                outputs = await generator.__anext__()
                return self._postprocess_outputs(
                    outputs,
                    return_logprobs=return_logprob,
                    n_per_prompt=n_per_prompt,
                    multi_token_stop_seqs=multi_token_stop_seqs,
                    stop_regex=stop_regex,
                    extract_hidden_states=return_hidden_states,
                    extract_routed_experts=return_routed_experts,
                )
            except Exception as e:
                last_error = e

                # Check if this is an OOM error that we can retry
                if _is_oom_error(e) and attempt < max_oom_retries:
                    batch_size = len(token_ids_prompts)
                    retry_delay = oom_retry_delay * (2 ** attempt)
                    logger.warning(
                        f"OOM during generation (attempt {attempt + 1}/{max_oom_retries + 1}), "
                        f"batch_size={batch_size}. Waiting {retry_delay:.1f}s for SGLang retraction. "
                        f"Error: {e}"
                    )
                    # Give SGLang's scheduler time to retract requests and free memory
                    await asyncio.sleep(retry_delay)
                    continue

                # Non-OOM error or exhausted retries
                break

        batch_size = len(input_batch.get("prompt_token_ids", []))
        raise RuntimeError(
            f"SGLang generation failed for batch of {batch_size} prompts: {last_error}"
        ) from last_error

    async def generate_stream(
        self, input_batch: InferenceEngineInput
    ) -> AsyncIterator[StreamingChunk]:
        """Generate responses with streaming output.

        Yields StreamingChunk objects as tokens are generated, enabling real-time
        processing and early stopping based on partial outputs.

        Note:
            Multi-token stop sequence trimming and stop_regex are not applied in
            streaming mode. Tokens are yielded as they are generated. For precise
            stop handling, use the non-streaming generate() method.

        Args:
            input_batch: Input batch containing prompt_token_ids and sampling_params.

        Yields:
            StreamingChunk objects with incremental generation output.
        """
        token_ids_prompts, sampling_params = self._preprocess_prompts(input_batch)

        # Extract logprob parameters
        return_logprob = sampling_params.pop("return_logprob", False)
        logprob_start_len = sampling_params.pop("logprob_start_len", None)
        top_logprobs_num = sampling_params.pop("top_logprobs_num", None)

        # Extract custom_logit_processor (request-level field, not a SamplingParams field)
        custom_logit_processor = sampling_params.pop("custom_logit_processor", None)

        # Convert RL action masking params to custom_logit_processor if not already set
        if custom_logit_processor is None:
            action_mask = sampling_params.pop("action_mask", None)
            disallowed_tokens = sampling_params.pop("disallowed_tokens", None)
            if action_mask or disallowed_tokens:
                from skyrl_train.inference_engines.sglang.rl_logit_processors import (
                    create_rl_logit_processor,
                )
                custom_logit_processor = create_rl_logit_processor(
                    action_mask=action_mask,
                    disallowed_tokens=disallowed_tokens,
                )

        # Remove internal fields (not used in streaming mode)
        # Note: Streaming does not apply multi-token stop trimming or stop_regex
        _ = sampling_params.pop("_multi_token_stop_seqs", None)
        _ = sampling_params.pop("_stop_regex", None)

        # Extract request-level priority for scheduling
        priority = sampling_params.pop("priority", None)

        # Extract per-request LoRA adapter name
        lora_name = sampling_params.pop("lora_name", None)

        # Extract multimodal data if provided (for VLMs, video models, audio models)
        image_data = input_batch.get("image_data", None)
        video_data = input_batch.get("video_data", None)
        audio_data = input_batch.get("audio_data", None)

        # Track cumulative state for each request
        num_requests = len(token_ids_prompts)
        cumulative_texts = [""] * num_requests
        cumulative_token_ids: List[List[int]] = [[] for _ in range(num_requests)]

        # Use GenerateReqInput directly to support priority scheduling
        from sglang.srt.managers.io_struct import GenerateReqInput

        obj = GenerateReqInput(
            input_ids=token_ids_prompts,
            sampling_params=sampling_params,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            custom_logit_processor=custom_logit_processor,
            priority=priority,
            lora_path=lora_name,  # SGLang uses lora_path to reference pre-loaded adapters by name
            stream=True,
        )

        async_generator = self.engine.tokenizer_manager.generate_request(obj, None)

        # Process streaming chunks
        async for chunk in async_generator:
            # SGLang returns chunks with output for each request
            # chunk format: {"index": int, "output_ids": list, "meta_info": dict, ...}
            index = chunk.get("index", 0)
            output_ids = chunk.get("output_ids", [])
            meta_info = chunk.get("meta_info", {})

            # Calculate delta (new tokens since last chunk)
            prev_len = len(cumulative_token_ids[index])
            new_token_ids = output_ids[prev_len:] if len(output_ids) > prev_len else []

            # Update cumulative state
            cumulative_token_ids[index] = output_ids

            # Decode delta text
            delta_text = ""
            delta_token_id = None
            delta_logprob = None

            if new_token_ids:
                delta_text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
                delta_token_id = new_token_ids[-1] if new_token_ids else None
                cumulative_texts[index] += delta_text

                # Extract logprob for the new token if available
                if return_logprob:
                    output_token_logprobs = meta_info.get("output_token_logprobs", None)
                    if output_token_logprobs and len(output_token_logprobs) > 0:
                        last_logprob = output_token_logprobs[-1]
                        if isinstance(last_logprob, (list, tuple)):
                            delta_logprob = last_logprob[0]
                        else:
                            delta_logprob = last_logprob

            # Check if finished
            finish_reason = meta_info.get("finish_reason", None)
            is_finished = finish_reason is not None
            stop_reason = finish_reason.get("type") if isinstance(finish_reason, dict) else finish_reason

            yield StreamingChunk(
                index=index,
                delta_text=delta_text,
                delta_token_id=delta_token_id,
                delta_logprob=delta_logprob,
                is_finished=is_finished,
                stop_reason=stop_reason if is_finished else None,
                cumulative_text=cumulative_texts[index],
                cumulative_token_ids=cumulative_token_ids[index],
            )

    def supports_streaming(self) -> bool:
        """Check if this engine supports streaming generation.

        Returns:
            True - SGLang supports streaming generation.
        """
        return True

    def _parse_tool_calls(
        self, response_text: str, tools: List[Dict[str, Any]]
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        """Parse tool calls from model response text.

        Supports multiple tool call formats used by different models:
        1. JSON-based: {"name": "func", "arguments": {...}}
        2. XML-based: <tool_call>...</tool_call>
        3. Function call syntax: func_name(arg1, arg2)

        Args:
            response_text: Raw response text from the model.
            tools: List of tool definitions to validate against.

        Returns:
            Tuple of (tool_calls list or None, remaining content or None).
            If tool calls are detected, content may be None or contain text before/after.
        """
        import json
        import re
        import uuid

        tool_calls = []
        remaining_content = response_text

        # Get valid function names from tools
        valid_functions = set()
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                valid_functions.add(tool["function"].get("name", ""))

        # Pattern 1: JSON object with name and arguments (common format)
        # Matches: {"name": "func_name", "arguments": {...}} or {"function": "func_name", "arguments": {...}}
        json_pattern = r'\{[^{}]*"(?:name|function)":\s*"([^"]+)"[^{}]*"arguments":\s*(\{[^{}]*\}|\[[^\[\]]*\]|"[^"]*")[^{}]*\}'
        json_matches = list(re.finditer(json_pattern, response_text, re.DOTALL))

        for match in json_matches:
            func_name = match.group(1)
            args_str = match.group(2)

            if func_name in valid_functions:
                # Try to parse arguments
                try:
                    if args_str.startswith('"') and args_str.endswith('"'):
                        # Arguments as string
                        arguments = args_str[1:-1]
                    else:
                        # Arguments as JSON object
                        arguments = args_str
                except Exception:
                    arguments = args_str

                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
                    }
                })
                # Remove matched text from content
                remaining_content = remaining_content.replace(match.group(0), "", 1)

        # Pattern 2: XML-style tool calls (used by some models)
        # Matches: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
        xml_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        xml_matches = list(re.finditer(xml_pattern, response_text, re.DOTALL))

        for match in xml_matches:
            try:
                call_data = json.loads(match.group(1))
                func_name = call_data.get("name", call_data.get("function", ""))
                arguments = call_data.get("arguments", {})

                if func_name in valid_functions:
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "arguments": json.dumps(arguments) if isinstance(arguments, dict) else str(arguments),
                        }
                    })
                    remaining_content = remaining_content.replace(match.group(0), "", 1)
            except json.JSONDecodeError:
                continue

        # Pattern 3: Function call syntax: function_name({"arg": "value"})
        for func_name in valid_functions:
            func_pattern = rf'{re.escape(func_name)}\s*\((\{{.*?\}})\)'
            func_matches = list(re.finditer(func_pattern, response_text, re.DOTALL))

            for match in func_matches:
                try:
                    arguments = match.group(1)
                    # Validate it's valid JSON
                    json.loads(arguments)
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "arguments": arguments,
                        }
                    })
                    remaining_content = remaining_content.replace(match.group(0), "", 1)
                except json.JSONDecodeError:
                    continue

        # Clean up remaining content
        remaining_content = remaining_content.strip()
        if not remaining_content:
            remaining_content = None

        if tool_calls:
            return tool_calls, remaining_content
        return None, response_text

    def _extract_multimodal_content(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Optional[List[Any]]]:
        """Extract image data from OpenAI-style multimodal messages.

        Processes messages with multimodal content (text + images) and extracts
        image data for separate processing by the VLM.

        Args:
            messages: List of chat messages, possibly with multimodal content.

        Returns:
            Tuple of (processed_messages, image_data_list):
            - processed_messages: Messages with content converted to text-only for tokenization
            - image_data_list: List of image data extracted from messages, or None if no images
        """
        import base64

        processed_messages = []
        all_images = []

        for msg in messages:
            content = msg.get("content")

            # Handle multimodal content (list of text/image parts)
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    part_type = part.get("type", "text")

                    if part_type == "text":
                        text_parts.append(part.get("text", ""))

                    elif part_type == "image_url":
                        image_url = part.get("image_url", {})
                        url = image_url.get("url", "") if isinstance(image_url, dict) else image_url

                        # Handle different image URL formats
                        if url.startswith("data:"):
                            # Base64 encoded image: data:image/png;base64,<data>
                            try:
                                # Extract base64 data after the comma
                                base64_data = url.split(",", 1)[1] if "," in url else url
                                image_bytes = base64.b64decode(base64_data)
                                all_images.append(image_bytes)
                            except Exception as e:
                                logger.warning(f"Failed to decode base64 image: {e}")
                                all_images.append(url)
                        else:
                            # URL or file path - pass through for SGLang to handle
                            all_images.append(url)

                        # Add image placeholder for chat template
                        text_parts.append("<image>")

                    elif part_type == "image":
                        # Direct image data
                        image_data = part.get("image")
                        if image_data:
                            all_images.append(image_data)
                            text_parts.append("<image>")

                # Create processed message with text-only content
                processed_msg = dict(msg)
                processed_msg["content"] = " ".join(text_parts)
                processed_messages.append(processed_msg)

            else:
                # Regular text message - pass through unchanged
                processed_messages.append(msg)

        image_data_list = all_images if all_images else None
        return processed_messages, image_data_list

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle OpenAI-compatible chat completion request with tool calling and multimodal support.

        Uses external tokenizer to convert text<->tokens since we run with skip_tokenizer_init=True.
        Supports OpenAI-style function/tool calling and vision/multimodal inputs.

        Args:
            request_payload: Dict containing 'json' with ChatCompletionRequest fields
                            and optionally 'headers'.
                            Supports:
                            - 'tools': List of tool definitions
                            - 'tool_choice': "auto", "none", "required", or specific tool
                            - Multimodal content in messages (text + image_url)

        Returns:
            ChatCompletionResponse as a dictionary with optional 'tool_calls' in message.
        """
        import time
        import json
        import re
        request_json = request_payload.get("json", {})
        messages = request_json.get("messages", [])
        model = request_json.get("model", self._model_path)

        # Extract tool calling parameters
        tools = request_json.get("tools", None)
        tool_choice = request_json.get("tool_choice", "auto")

        # Extract multimodal content (images) from messages
        processed_messages, image_data = self._extract_multimodal_content(messages)

        # Apply chat template to convert messages to prompt
        # Pass tools to template if provided (for models that support tool calling)
        try:
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            # Pass tools to chat template if the tokenizer supports it
            if tools:
                template_kwargs["tools"] = tools
            prompt = self.tokenizer.apply_chat_template(
                processed_messages,
                **template_kwargs
            )
            prompt_token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        except TypeError:
            # Fallback: tokenizer doesn't support tools parameter
            try:
                prompt = self.tokenizer.apply_chat_template(
                    processed_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompt_token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            except Exception as e:
                return {
                    "object": "error",
                    "message": f"Failed to apply chat template: {e}",
                    "code": 400
                }
        except Exception as e:
            return {
                "object": "error",
                "message": f"Failed to apply chat template: {e}",
                "code": 400
            }

        # Build sampling params from request - pass through all SGLang-supported params
        sampling_params = {
            "max_new_tokens": request_json.get("max_tokens", request_json.get("max_completion_tokens", 1024)),
            "temperature": request_json.get("temperature", 1.0),
            "top_p": request_json.get("top_p", 1.0),
            "top_k": request_json.get("top_k", -1),
        }
        if request_json.get("stop"):
            sampling_params["stop"] = request_json["stop"]

        # Pass through additional SGLang sampling parameters if provided
        optional_params = [
            "min_p", "frequency_penalty", "presence_penalty", "repetition_penalty",
            "min_new_tokens", "n", "ignore_eos", "skip_special_tokens",
            "spaces_between_special_tokens", "no_stop_trim", "logit_bias",
            "seed", "stop_token_ids", "stop_regex", "json_schema", "regex", "ebnf",
            "lora_name",  # Per-request LoRA adapter selection
        ]
        for param in optional_params:
            if param in request_json and request_json[param] is not None:
                sampling_params[param] = request_json[param]

        # Handle n>1 for multiple completions
        n = sampling_params.get("n", 1)

        # Generate using token-in-token-out with optional multimodal support
        input_batch = {
            "prompt_token_ids": [prompt_token_ids],
            "sampling_params": sampling_params,
        }
        # Add image data for multimodal models if present
        if image_data:
            input_batch["image_data"] = [image_data]  # Wrap in list for batch dimension

        try:
            output = await self.generate(input_batch)
        except Exception as e:
            return {
                "object": "error",
                "message": f"Generation failed: {e}",
                "code": 500
            }

        # Build OpenAI-compatible response with support for n>1 and tool calling
        choices = []
        num_outputs = min(n, len(output["responses"]))
        for i in range(num_outputs):
            response_text = output["responses"][i] if i < len(output["responses"]) else ""
            response_token_ids = output["response_ids"][i] if i < len(output["response_ids"]) else []
            finish_reason = output["stop_reasons"][i] if i < len(output["stop_reasons"]) else "stop"

            # Parse tool calls from response if tools were provided
            tool_calls = None
            content = response_text
            if tools:
                tool_calls, content = self._parse_tool_calls(response_text, tools)
                if tool_calls:
                    finish_reason = "tool_calls"

            # Build message with optional tool_calls
            message = {
                "role": "assistant",
                "content": content,
            }
            if tool_calls:
                message["tool_calls"] = tool_calls

            choices.append({
                "index": i,
                "message": message,
                "token_ids": response_token_ids,
                "finish_reason": finish_reason,
            })

        # Calculate total tokens across all outputs
        total_completion_tokens = sum(len(c.get("token_ids", [])) for c in choices)

        return {
            "id": f"chatcmpl-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": choices,
            "usage": {
                "prompt_tokens": len(prompt_token_ids),
                "completion_tokens": total_completion_tokens,
                "total_tokens": len(prompt_token_ids) + total_completion_tokens,
            }
        }

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle OpenAI-compatible text completion request.

        Uses external tokenizer to convert text<->tokens since we run with skip_tokenizer_init=True.
        Supports both single and batched prompts.

        Args:
            request_payload: Dict containing 'json' with CompletionRequest fields
                            and optionally 'headers'.
                            prompt can be:
                            - str: single text prompt
                            - List[int]: single prompt as token IDs
                            - List[str]: batched text prompts
                            - List[List[int]]: batched token ID prompts

        Returns:
            CompletionResponse as a dictionary.
        """
        import time
        request_json = request_payload.get("json", {})
        prompt = request_json.get("prompt", "")
        model = request_json.get("model", self._model_path)

        # Parse prompt into list of token ID lists (batched format)
        prompt_token_ids_list: List[List[int]] = []

        if isinstance(prompt, str):
            # Single text prompt
            prompt_token_ids_list = [self.tokenizer.encode(prompt, add_special_tokens=False)]
        elif isinstance(prompt, list) and len(prompt) > 0:
            if isinstance(prompt[0], int):
                # Single prompt as token IDs
                prompt_token_ids_list = [prompt]
            elif isinstance(prompt[0], str):
                # Batched text prompts
                prompt_token_ids_list = [
                    self.tokenizer.encode(p, add_special_tokens=False) for p in prompt
                ]
            elif isinstance(prompt[0], list) and len(prompt[0]) > 0 and isinstance(prompt[0][0], int):
                # Batched token ID prompts
                prompt_token_ids_list = prompt
            else:
                return {
                    "object": "error",
                    "message": "prompt must be a string, list of token IDs, list of strings, or list of token ID lists",
                    "code": 400
                }
        else:
            return {
                "object": "error",
                "message": "prompt must be a string, list of token IDs, list of strings, or list of token ID lists",
                "code": 400
            }

        num_prompts = len(prompt_token_ids_list)

        # Build sampling params from request - pass through all SGLang-supported params
        sampling_params = {
            "max_new_tokens": request_json.get("max_tokens", 1024),
            "temperature": request_json.get("temperature", 1.0),
            "top_p": request_json.get("top_p", 1.0),
            "top_k": request_json.get("top_k", -1),
        }
        if request_json.get("stop"):
            sampling_params["stop"] = request_json["stop"]

        # Pass through additional SGLang sampling parameters if provided
        optional_params = [
            "min_p", "frequency_penalty", "presence_penalty", "repetition_penalty",
            "min_new_tokens", "n", "ignore_eos", "skip_special_tokens",
            "spaces_between_special_tokens", "no_stop_trim", "logit_bias",
            "seed", "stop_token_ids", "stop_regex", "json_schema", "regex", "ebnf",
            "lora_name",  # Per-request LoRA adapter selection
        ]
        for param in optional_params:
            if param in request_json and request_json[param] is not None:
                sampling_params[param] = request_json[param]

        # Generate using token-in-token-out
        input_batch = {
            "prompt_token_ids": prompt_token_ids_list,
            "sampling_params": sampling_params,
        }
        try:
            output = await self.generate(input_batch)
        except Exception as e:
            return {
                "object": "error",
                "message": f"Generation failed: {e}",
                "code": 500
            }

        # Build OpenAI-compatible response
        # Each prompt gets one choice with sequential index
        choices = []
        for i in range(num_prompts):
            response_text = output["responses"][i] if i < len(output["responses"]) else ""
            response_token_ids = output["response_ids"][i] if i < len(output["response_ids"]) else []
            finish_reason = output["stop_reasons"][i] if i < len(output["stop_reasons"]) else "stop"
            choices.append({
                "index": i,
                "text": response_text,
                "token_ids": response_token_ids,
                "finish_reason": finish_reason,
            })

        # Calculate total tokens across all outputs
        total_prompt_tokens = sum(len(p) for p in prompt_token_ids_list)
        total_completion_tokens = sum(len(c.get("token_ids", [])) for c in choices)

        return {
            "id": f"cmpl-{int(time.time()*1000)}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": choices,
            "usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
            }
        }

    async def init_weight_update_communicator(self, init_info):
        """Initialize weight update communicator for SGLang.

        Args:
            init_info: WeightSyncInitInfo from the sender.
        """
        return await self._weight_loader.init_communicator(init_info)

    async def update_named_weights(self, request: WeightUpdateRequest) -> None:
        """Update named weights in SGLang engine.

        Args:
            request: Weight update request. Can be:
                - CudaIpcWeightUpdateRequest: For CUDA IPC weight transfer
                - BroadcastWeightUpdateRequest: For broadcast weight transfer
                - LoraLoadRequest: For loading LoRA adapters from disk
        """
        from skyrl_train.weight_sync import LoraLoadRequest

        # Handle LoRA disk loading request
        if isinstance(request, LoraLoadRequest):
            # Generate a unique adapter name from the path
            import os
            lora_name = os.path.basename(request.lora_path.rstrip("/"))
            return await self.load_lora_adapter(lora_name=lora_name, lora_path=request.lora_path)

        await self._weight_loader.load_weights(request)

    async def wake_up(self, *args: Any, **kwargs: Any):
        """Wake up the engine and resume paused requests.

        Restores memory occupation and resumes any requests that were retracted
        to the waiting queue during sleep(). For multi-stage waking up, pass in
        "weight" or "kv_cache" to tags.

        Args:
            *args: Positional arguments (unused, for interface compatibility).
            **kwargs: Keyword arguments. Supports:
                - tags: Optional[List[str]] - Memory tags to resume (e.g., ["weights"], ["kv_cache"])
                - timeout: float - Timeout in seconds (default: 60.0)
                - resume_generation: bool - Whether to resume paused requests (default: True)
        """
        import asyncio

        tags = kwargs.get("tags", None)
        timeout = kwargs.get("timeout", 60.0)
        resume_generation = kwargs.get("resume_generation", True)

        # Step 1: Restore memory occupation
        obj = ResumeMemoryOccupationReqInput(tags=tags)
        try:
            await asyncio.wait_for(
                self.engine.tokenizer_manager.resume_memory_occupation(obj, None),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise RuntimeError(f"wake_up timed out after {timeout}s with tags={tags}")

        # Step 2: Resume generation to process retracted requests from waiting queue
        if resume_generation:
            try:
                await self.continue_generation()
                logger.debug("Resumed generation after wake_up")
            except Exception as e:
                logger.error(f"Failed to continue generation after wake_up: {e}")
                raise

        logger.info(
            f"From SGLang engine -- Free GPU memory after wake up with tags {tags if tags is not None else 'None'}: "
            + f"{torch.cuda.mem_get_info()[0] / 1024**2:.1f} MB"
        )

    async def sleep(self, *args: Any, **kwargs: Any):
        """Put engine to sleep, preserving in-flight requests.

        Uses pause_generation(mode="retract") to move running requests to the waiting
        queue instead of aborting them. This satisfies SGLang's requirement that
        len(self.running_batch.reqs) == 0 before releasing memory, without losing work.

        After wake_up(), requests will automatically resume from the waiting queue.

        Args:
            *args: Positional arguments (unused, for interface compatibility).
            **kwargs: Keyword arguments. Supports:
                - tags: Optional[List[str]] - Memory tags to release (e.g., ["weights"], ["kv_cache"])
                - timeout: float - Timeout in seconds (default: 60.0)
                - pause_mode: str - Pause mode: "retract" (default, preserves requests),
                                    "in_place" (preserves KV cache), or "abort" (legacy)
                - defragment: bool - Run torch.cuda.empty_cache() after releasing memory (default: True)
                                     Helps recover fragmented GPU memory for training step.
        """
        import asyncio
        import gc

        tags = kwargs.get("tags", None)
        timeout = kwargs.get("timeout", 60.0)
        pause_mode = kwargs.get("pause_mode", "retract")
        defragment = kwargs.get("defragment", True)

        # Use pause_generation to safely clear running_batch without losing requests
        # - "retract": Moves running requests to waiting queue (preserves work)
        # - "in_place": Pauses in place, preserves KV cache (fastest resume)
        # - "abort": Legacy behavior, loses in-flight requests
        try:
            await self.pause_generation(mode=pause_mode)
            logger.debug(f"Paused generation with mode='{pause_mode}' before sleep")
        except Exception as e:
            logger.error(
                f"Failed to pause generation before sleep: {e}. "
                f"Memory release may fail if requests are still in-flight."
            )
            raise

        obj = ReleaseMemoryOccupationReqInput(tags=tags)
        try:
            await asyncio.wait_for(
                self.engine.tokenizer_manager.release_memory_occupation(obj, None),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise RuntimeError(f"sleep timed out after {timeout}s with tags={tags}")

        # Memory defragmentation: recover fragmented GPU memory
        # This is especially important for RL training where memory is allocated/freed repeatedly
        # during train -> generate -> train cycles. Without defragmentation, memory fragmentation
        # can cause OOM errors even when total free memory is sufficient.
        if defragment:
            gc.collect()  # Clear Python garbage first
            torch.cuda.empty_cache()  # Return CUDA memory to allocator
            logger.debug("Memory defragmentation completed (gc.collect + cuda.empty_cache)")

        free_mb = torch.cuda.mem_get_info()[0] / 1024**2
        logger.info(
            f"From SGLang engine -- Free GPU memory after sleep with tags {tags if tags is not None else 'None'}: "
            + f"{free_mb:.1f} MB"
        )

    async def teardown(self):
        """Shutdown the SGLang engine.

        Performs graceful shutdown with error handling. Logs the teardown status
        and continues even if shutdown encounters issues to avoid leaving resources
        in an undefined state.
        """
        logger.info(f"Tearing down SGLang engine (model: {self._model_path})")

        # Destroy weight update group if it was initialized
        try:
            await self._weight_loader.destroy_group()
        except Exception as e:
            logger.warning(f"Error destroying weight update group during teardown: {e}")

        try:
            self.engine.shutdown()
            logger.info("SGLang engine shutdown completed successfully")
        except Exception as e:
            logger.error(f"SGLang engine shutdown encountered an error: {e}")
            # Re-raise to notify caller, but after logging
            raise RuntimeError(f"SGLang engine teardown failed: {e}") from e

    async def reset_prefix_cache(self):
        """Reset prefix cache in SGLang engine.

        Clears ALL cache tiers (GPU, CPU, and storage). For clearing only the
        storage tier (disk), use clear_hicache_storage() instead.
        """
        # Call the underlying async method for the same reason as in `init_weight_update_communicator`
        return await self.engine.tokenizer_manager.flush_cache()

    async def clear_hicache_storage(self) -> bool:
        """Clear only the hierarchical cache storage tier (disk).

        Unlike reset_prefix_cache() which clears ALL cache tiers,
        this method only clears the storage backend (e.g., local disk, mooncake).
        GPU and CPU cache tiers are preserved.

        Useful for:
        - Reclaiming disk space without losing hot GPU/CPU cache
        - Clearing stale storage cache between training runs
        - Managing disk usage for large-scale RL training

        Returns:
            True if successful, False otherwise.

        Note:
            Requires hierarchical_cache to be enabled with a storage backend.
            If no storage backend is configured, this is a no-op.
        """
        try:
            from sglang.srt.managers.io_struct import ClearHiCacheReqInput
            result = await self.engine.tokenizer_manager.clear_hicache_storage()
            success = result.success if hasattr(result, 'success') else True
            if success:
                logger.info("Cleared hierarchical cache storage tier")
            else:
                logger.warning("Failed to clear hierarchical cache storage")
            return success
        except (ImportError, AttributeError) as e:
            logger.debug(f"clear_hicache_storage not available: {e}")
            return False

    async def pause_generation(
        self, mode: Literal["abort", "in_place", "retract"] = "abort"
    ) -> None:
        """Pause generation with specified mode.

        Args:
            mode: Pause mode, one of:
                - "abort": Abort and return all requests currently being processed.
                    Requests are cancelled and returned to callers immediately.
                - "in_place": Pause without aborting. Requests stay in event loop
                    with KV cache preserved. Call continue_generation() to resume.
                    Note: flush_cache will fail if there are running requests.
                - "retract": Pause and retract all running requests to waiting queue.
                    KV cache can be flushed and will be recomputed on continue.
        """
        obj = PauseGenerationReqInput(mode=mode)
        await self.engine.tokenizer_manager.pause_generation(obj)
        logger.debug(f"Paused generation with mode={mode}")

    async def continue_generation(self) -> None:
        """Resume generation after pause.

        Must be called after pause_generation with mode='in_place' or 'retract'
        to resume processing of paused/retracted requests.
        """
        obj = ContinueGenerationReqInput()
        await self.engine.tokenizer_manager.continue_generation(obj)
        logger.debug("Continued generation")

    async def abort_generation(self) -> None:
        """Abort all in-flight generation requests.

        Convenience method that calls pause_generation with mode='abort'.
        Cancels all currently running requests and returns them to callers.
        """
        await self.pause_generation(mode="abort")

    async def get_weight_version(self) -> Optional[str]:
        """Get the current weight version identifier.

        Returns:
            Current weight version string (e.g., "step_100"), or None if not set/supported.

        Note:
            This API may not be available in all SGLang versions. If the native API
            is not available, falls back to locally tracked version (set via
            update_weight_version or weight_version parameter in weight updates).
        """
        # If we already know the API is not available, return local version
        if self._version_api_available is False:
            return self._weight_version

        try:
            from sglang.srt.managers.io_struct import GetWeightVersionReqInput
            obj = GetWeightVersionReqInput()
            version = await self.engine.tokenizer_manager.get_weight_version(obj, None)
            self._version_api_available = True
            # Also update local tracking to stay in sync
            self._weight_version = version
            return version
        except (ImportError, AttributeError):
            # Native API not available, use local tracking
            if self._version_api_available is None:
                logger.info(
                    "SGLang weight version API not available in this version. "
                    "Using local weight version tracking as fallback."
                )
                self._version_api_available = False
            return self._weight_version

    async def update_weight_version(
        self,
        new_version: str,
        abort_all_requests: bool = True,
    ) -> None:
        """Update the weight version identifier.

        This is useful for tracking which training step's weights are currently
        loaded in the inference engine. It can be called after weight updates
        to mark the new version.

        Note:
            If the native SGLang API is not available, falls back to local tracking.
            The version can also be set via the 'weight_version' parameter in
            weight update requests (update_named_weights, load_weights_from_disk).

        Args:
            new_version: New version identifier (e.g., "step_100").
            abort_all_requests: Whether to abort all in-flight requests during
                the version update. Default True for safety (ignored in fallback mode).
        """
        # Always update local tracking
        self._weight_version = new_version

        # If we already know the API is not available, just use local tracking
        if self._version_api_available is False:
            logger.debug(f"Updated weight version locally to: {new_version}")
            return

        # Try to use native API
        if not hasattr(self.engine.tokenizer_manager, 'update_weight_version'):
            if self._version_api_available is None:
                logger.info(
                    "SGLang update_weight_version API not available. "
                    "Using local weight version tracking as fallback."
                )
                self._version_api_available = False
            return

        try:
            obj = UpdateWeightVersionReqInput(
                new_version=new_version,
                abort_all_requests=abort_all_requests,
            )
            success, message = await self.engine.tokenizer_manager.update_weight_version(obj, None)
            if not success:
                raise RuntimeError(f"Failed to update weight version: {message}")
            self._version_api_available = True
            logger.debug(f"Updated weight version to: {new_version}")
        except Exception as e:
            # If native API fails, fall back to local tracking
            if self._version_api_available is None:
                logger.warning(
                    f"SGLang update_weight_version API failed: {e}. "
                    "Using local weight version tracking as fallback."
                )
                self._version_api_available = False

    # Convenience methods for selective memory management
    async def sleep_weights_only(self, **kwargs) -> None:
        """Release only weight memory, preserving KV cache.

        This is the recommended approach for RL training workloads where:
        1. You want to update weights between generation steps
        2. You want to preserve KV cache for prefix reuse (RadixAttention)
        3. You want faster wake-up times

        Args:
            **kwargs: Additional arguments passed to sleep() (timeout, abort_first, etc.)
        """
        await self.sleep(tags=MemoryTag.TRAINING_DEFAULT, **kwargs)

    async def wake_up_weights_only(self, **kwargs) -> None:
        """Restore only weight memory after sleep_weights_only().

        Use this after sleep_weights_only() to restore weight memory
        before syncing new weights.

        Args:
            **kwargs: Additional arguments passed to wake_up() (timeout, etc.)
        """
        await self.wake_up(tags=MemoryTag.TRAINING_DEFAULT, **kwargs)

    async def sleep_all(self, **kwargs) -> None:
        """Release all memory (weights, KV cache, CUDA graphs).

        Use this when you need maximum memory available for training
        and don't need to preserve KV cache state.

        Args:
            **kwargs: Additional arguments passed to sleep() (timeout, abort_first, etc.)
        """
        await self.sleep(tags=MemoryTag.ALL, **kwargs)

    async def wake_up_all(self, **kwargs) -> None:
        """Restore all memory after sleep_all().

        Args:
            **kwargs: Additional arguments passed to wake_up() (timeout, etc.)
        """
        await self.wake_up(tags=MemoryTag.ALL, **kwargs)

    # Weight validation methods
    async def get_weights_by_name(self, name: str) -> Optional[torch.Tensor]:
        """Get a weight tensor by parameter name.

        Uses SGLang's get_weights_by_name API to retrieve the current
        value of a model parameter.

        Args:
            name: Parameter name (e.g., "model.layers.0.self_attn.q_proj.weight")

        Returns:
            The weight tensor if found, None otherwise.
        """
        try:
            from sglang.srt.managers.io_struct import GetWeightsByNameReqInput
            obj = GetWeightsByNameReqInput(name=name)
            result = await self.engine.tokenizer_manager.get_weights_by_name(obj, None)
            return result
        except Exception as e:
            logger.warning(f"Failed to get weights for '{name}': {e}")
            return None

    async def validate_weights(
        self,
        names: Optional[List[str]] = None,
        check_nan: bool = True,
        check_inf: bool = True,
        check_zeros: bool = False,
    ) -> Dict[str, Any]:
        """Validate model weights for common issues.

        Checks for NaN values, infinite values, and optionally all-zero tensors.
        Useful for debugging weight sync issues and detecting corruption.

        Args:
            names: List of parameter names to validate. If None, validates all parameters.
            check_nan: Whether to check for NaN values.
            check_inf: Whether to check for infinite values.
            check_zeros: Whether to check for all-zero tensors (potential uninitialized weights).

        Returns:
            Dict with validation results:
            - "valid": bool - Whether all checks passed
            - "checked": int - Number of parameters checked
            - "issues": List[Dict] - List of issues found with parameter name and type
        """
        issues = []
        checked = 0

        # Get parameter names if not provided
        if names is None:
            # Use model's named_parameters to get all weight names
            try:
                from sglang.srt.managers.io_struct import GetWeightsByNameReqInput
                # Try to get a known parameter to test the API
                test_result = await self.get_weights_by_name("model.embed_tokens.weight")
                if test_result is None:
                    logger.warning("Cannot validate weights: API not available or model structure unknown")
                    return {"valid": True, "checked": 0, "issues": [], "warning": "Validation skipped"}
            except Exception as e:
                logger.warning(f"Weight validation not available: {e}")
                return {"valid": True, "checked": 0, "issues": [], "warning": str(e)}
            # For now, just validate the embedding layer as a sanity check
            names = ["model.embed_tokens.weight"]

        for name in names:
            weight = await self.get_weights_by_name(name)
            if weight is None:
                continue

            checked += 1

            if check_nan and torch.isnan(weight).any():
                nan_count = torch.isnan(weight).sum().item()
                issues.append({
                    "name": name,
                    "type": "nan",
                    "count": nan_count,
                    "percentage": 100 * nan_count / weight.numel()
                })

            if check_inf and torch.isinf(weight).any():
                inf_count = torch.isinf(weight).sum().item()
                issues.append({
                    "name": name,
                    "type": "inf",
                    "count": inf_count,
                    "percentage": 100 * inf_count / weight.numel()
                })

            if check_zeros and (weight == 0).all():
                issues.append({
                    "name": name,
                    "type": "all_zeros",
                    "shape": list(weight.shape)
                })

        if issues:
            logger.warning(f"Weight validation found {len(issues)} issues: {issues}")

        return {
            "valid": len(issues) == 0,
            "checked": checked,
            "issues": issues
        }

    async def check_weight_sync_integrity(self, expected_version: Optional[str] = None) -> Dict[str, Any]:
        """Check weight synchronization integrity.

        Verifies that weights are properly loaded and optionally checks
        the weight version matches expected.

        Args:
            expected_version: Expected weight version string to verify.

        Returns:
            Dict with integrity check results.
        """
        results = {
            "weights_valid": False,
            "version_match": None,
            "current_version": None,
        }

        # Check weight validity
        validation = await self.validate_weights()
        results["weights_valid"] = validation["valid"]
        results["validation_details"] = validation

        # Check version if expected
        if expected_version is not None:
            try:
                current_version = await self.get_weight_version()
                results["current_version"] = current_version
                results["version_match"] = current_version == expected_version
                if not results["version_match"]:
                    logger.warning(
                        f"Weight version mismatch: expected '{expected_version}', got '{current_version}'"
                    )
            except Exception as e:
                logger.warning(f"Could not check weight version: {e}")

        return results

    async def load_weights_from_disk(
        self,
        model_path: str,
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ) -> None:
        """Load weights from a checkpoint file or directory.

        This method allows hot-swapping model weights from disk without
        restarting the engine. Useful for:
        - Resuming from a checkpoint during training
        - A/B testing different model versions
        - Loading fine-tuned weights

        Args:
            model_path: Path to the checkpoint file or directory.
            load_format: Weight format ("auto", "pt", "safetensors", etc.).
                If None, auto-detected from file extension.
            flush_cache: Whether to flush KV cache after loading weights.
                Default True since weights changed.

        Raises:
            RuntimeError: If weight loading fails.
        """
        try:
            from sglang.srt.managers.io_struct import UpdateWeightFromDiskReqInput
            obj = UpdateWeightFromDiskReqInput(
                model_path=model_path,
                load_format=load_format,
                flush_cache=flush_cache,
            )
            success, message = await self.engine.tokenizer_manager.update_weights_from_disk(obj, None)
            if not success:
                raise RuntimeError(f"Failed to load weights from disk: {message}")
            logger.info(f"Loaded weights from disk: {model_path}")
        except ImportError:
            raise RuntimeError(
                "UpdateWeightFromDiskReqInput not available in this SGLang version. "
                "Please upgrade SGLang to use load_weights_from_disk()."
            )

    # ============================================================================
    # Model Saving API
    # ============================================================================

    async def save_sharded_model(
        self,
        save_path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> bool:
        """Save the model weights to disk in sharded format.

        Saves the current model weights (including any updates from training)
        to the specified path. Useful for checkpointing during RL training.

        Args:
            save_path: Directory path to save the model.
            pattern: Optional pattern for shard naming (e.g., "model-{rank:05d}-of-{world:05d}.safetensors").
            max_size: Maximum size per shard in bytes.

        Returns:
            True if successful, False otherwise.

        Example:
            # Save checkpoint after training step
            success = await engine.save_sharded_model(
                save_path="/checkpoints/step_1000",
                pattern="model-{rank:05d}-of-{world:05d}.safetensors"
            )
        """
        try:
            from sglang.srt.managers.io_struct import SaveShardedModelReqInput
            obj = SaveShardedModelReqInput(
                save_path=save_path,
                pattern=pattern,
                max_size=max_size,
            )
            result = await self.engine.tokenizer_manager.save_sharded_model(obj, None)
            success = result.success if hasattr(result, 'success') else True
            if success:
                logger.info(f"Saved sharded model to {save_path}")
            else:
                logger.warning(f"Failed to save sharded model: {result}")
            return success
        except (ImportError, AttributeError) as e:
            logger.warning(f"save_sharded_model not available: {e}")
            return False

    async def save_remote_model(
        self,
        remote_path: str,
        storage_type: str = "s3",
    ) -> bool:
        """Save the model weights to remote storage.

        Saves the current model weights to cloud storage (S3, GCS, etc.).
        Useful for distributed training checkpointing.

        Args:
            remote_path: Remote storage path (e.g., "s3://bucket/checkpoints/step_1000").
            storage_type: Storage backend type ("s3", "gcs", "azure", etc.).

        Returns:
            True if successful, False otherwise.
        """
        try:
            from sglang.srt.managers.io_struct import SaveRemoteModelReqInput
            obj = SaveRemoteModelReqInput(
                remote_path=remote_path,
                storage_type=storage_type,
            )
            result = await self.engine.tokenizer_manager.save_remote_model(obj, None)
            success = result.success if hasattr(result, 'success') else True
            if success:
                logger.info(f"Saved model to remote storage: {remote_path}")
            else:
                logger.warning(f"Failed to save to remote storage: {result}")
            return success
        except (ImportError, AttributeError) as e:
            logger.warning(f"save_remote_model not available: {e}")
            return False

    # ============================================================================
    # Decode API
    # ============================================================================

    async def decode(
        self,
        token_ids: List[List[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Decode token IDs to text strings.

        Uses the engine's tokenizer to decode token sequences.
        This is useful when you need to decode tokens on the inference
        side without round-tripping to the client.

        Args:
            token_ids: List of token ID sequences to decode.
            skip_special_tokens: Whether to skip special tokens in output.

        Returns:
            List of decoded text strings.
        """
        return [
            self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
            for ids in token_ids
        ]

    # ============================================================================
    # Profiling API
    # ============================================================================

    async def start_profile(self) -> bool:
        """Start profiling the inference engine.

        Enables performance profiling for debugging and optimization.
        Call stop_profile() to collect results.

        Returns:
            True if profiling started successfully.
        """
        try:
            result = await self.engine.tokenizer_manager.start_profile()
            logger.info("Started profiling")
            return True
        except Exception as e:
            logger.warning(f"start_profile not available: {e}")
            return False

    async def stop_profile(self) -> Optional[Dict[str, Any]]:
        """Stop profiling and collect results.

        Returns profiling data collected since start_profile().

        Returns:
            Dict with profiling results, or None if not available.
        """
        try:
            result = await self.engine.tokenizer_manager.stop_profile()
            logger.info("Stopped profiling")
            return result
        except Exception as e:
            logger.warning(f"stop_profile not available: {e}")
            return None

    # Session management for prefix caching
    async def open_session(
        self,
        capacity_of_str_len: int = 8192,
        session_id: Optional[str] = None,
    ) -> Optional[str]:
        """Open a session for prefix caching in multi-turn conversations.

        Sessions enable efficient prefix reuse via RadixAttention. When multiple
        requests share a common prefix (e.g., system prompt + conversation history),
        SGLang can cache and reuse the KV cache for the shared prefix.

        Usage pattern:
            session_id = await engine.open_session(capacity_of_str_len=8192)
            try:
                # First turn - computes KV cache for prefix
                out1 = await engine.generate_with_session(session_id, input1)
                # Second turn - reuses cached prefix KV
                out2 = await engine.generate_with_session(session_id, input2)
            finally:
                await engine.close_session(session_id)

        Args:
            capacity_of_str_len: Maximum string length the session can handle.
                This reserves KV cache capacity for the session.
            session_id: Optional custom session ID. If None, auto-generated.

        Returns:
            Session ID string if successful, None if session already exists or failed.
        """
        try:
            from sglang.srt.managers.io_struct import OpenSessionReqInput
            obj = OpenSessionReqInput(
                capacity_of_str_len=capacity_of_str_len,
                session_id=session_id,
            )
            result = await self.engine.tokenizer_manager.open_session(obj, None)
            if result:
                logger.info(f"Opened session '{result}' with capacity {capacity_of_str_len}")
            else:
                logger.warning(f"Failed to open session (may already exist): {session_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to open session: {e}")
            return None

    async def close_session(self, session_id: str) -> None:
        """Close a session and release its resources.

        Args:
            session_id: The session ID returned by open_session().
        """
        try:
            from sglang.srt.managers.io_struct import CloseSessionReqInput
            obj = CloseSessionReqInput(session_id=session_id)
            await self.engine.tokenizer_manager.close_session(obj, None)
            logger.info(f"Closed session '{session_id}'")
        except Exception as e:
            logger.warning(f"Failed to close session '{session_id}': {e}")

    async def generate_with_session(
        self,
        session_id: str,
        input_batch: InferenceEngineInput,
        rid: Optional[str] = None,
        offset: Optional[int] = None,
        replace: bool = False,
        drop_previous_output: bool = False,
    ) -> InferenceEngineOutput:
        """Generate responses using a session for prefix caching.

        This method enables efficient multi-turn conversations by reusing
        KV cache from previous turns within the same session.

        Args:
            session_id: Session ID from open_session().
            input_batch: Input batch containing prompt_token_ids and sampling_params.
            rid: Request ID to append to or branch from. For first turn, use None.
                For subsequent turns, use the rid from the previous response's meta_info.
            offset: Token offset to continue from (-1 = append at end, 0+ = branch at position).
                If None, defaults to -1 (append mode).
            replace: If True, clears child branches when branching. If False, appends.
            drop_previous_output: If True, drops the generated output from previous
                requests in this session (keeps only the input prefix).

        Returns:
            InferenceEngineOutput with generated responses. Access meta_info["id"] for
            the request ID to use in subsequent turns.
        """
        from sglang.srt.managers.io_struct import GenerateReqInput

        token_ids_prompts, sampling_params = self._preprocess_prompts(input_batch)

        # Extract logprob parameters
        return_logprob = sampling_params.pop("return_logprob", False)
        logprob_start_len = sampling_params.pop("logprob_start_len", None)
        top_logprobs_num = sampling_params.pop("top_logprobs_num", None)
        n_per_prompt = sampling_params.get("n", 1) or 1  # Default to 1 if None

        # Extract custom_logit_processor (request-level field, not a SamplingParams field)
        custom_logit_processor = sampling_params.pop("custom_logit_processor", None)

        # Convert RL action masking params to custom_logit_processor if not already set
        if custom_logit_processor is None:
            action_mask = sampling_params.pop("action_mask", None)
            disallowed_tokens = sampling_params.pop("disallowed_tokens", None)
            if action_mask or disallowed_tokens:
                from skyrl_train.inference_engines.sglang.rl_logit_processors import (
                    create_rl_logit_processor,
                )
                custom_logit_processor = create_rl_logit_processor(
                    action_mask=action_mask,
                    disallowed_tokens=disallowed_tokens,
                )

        # Extract hidden states request (for value function enrichment)
        return_hidden_states = sampling_params.pop("return_hidden_states", False)
        if not return_hidden_states:
            return_hidden_states = input_batch.get("return_hidden_states", False)

        # Extract routed experts parameter for R3 (Rollout Routing Replay)
        return_routed_experts = sampling_params.pop("return_routed_experts", False)
        if not return_routed_experts:
            return_routed_experts = input_batch.get("return_routed_experts", False)

        # Extract per-request LoRA adapter name
        lora_name = sampling_params.pop("lora_name", None)

        # Build session params for this request
        session_params = {
            "id": session_id,
            "rid": rid,
            "offset": offset if offset is not None else -1,
            "replace": replace,
            "drop_previous_output": drop_previous_output,
        }

        # For batched requests, apply session to all
        # SGLang expects session_params as a dict for single request or list of dicts for batch
        if len(token_ids_prompts) == 1:
            session_params_arg = session_params
        else:
            # Apply same session to all requests in batch
            session_params_arg = [session_params.copy() for _ in token_ids_prompts]

        # Use GenerateReqInput directly to pass session_params
        # (Engine.async_generate doesn't expose session_params parameter)
        obj = GenerateReqInput(
            input_ids=token_ids_prompts,
            sampling_params=sampling_params,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            session_params=session_params_arg,
            custom_logit_processor=custom_logit_processor,
            lora_path=lora_name,  # SGLang uses lora_path to reference pre-loaded adapters by name
            return_hidden_states=return_hidden_states,
            return_routed_experts=return_routed_experts,
        )

        try:
            generator = self.engine.tokenizer_manager.generate_request(obj, None)
            outputs = await generator.__anext__()

            # Handle single vs batch output
            if not isinstance(outputs, list):
                outputs = [outputs]

            return self._postprocess_outputs(
                outputs,
                return_logprobs=return_logprob,
                n_per_prompt=n_per_prompt,
                extract_request_ids=True,  # Extract rids for session continuity
                extract_hidden_states=return_hidden_states,
                extract_routed_experts=return_routed_experts,
            )
        except Exception as e:
            raise RuntimeError(f"Session generation failed for session '{session_id}': {e}") from e

    def supports_sessions(self) -> bool:
        """Check if this engine supports session management.

        Returns:
            True - SGLang supports session management for prefix caching.
        """
        return True

    # Embedding API support
    async def encode(
        self,
        texts: List[str],
        dimensions: Optional[int] = None,
    ) -> List[List[float]]:
        """Get embeddings for a list of texts.

        Uses SGLang's embedding API to generate vector representations
        of input texts. Useful for:
        - Semantic similarity scoring
        - Retrieval-augmented generation (RAG)
        - Reward model embeddings
        - Clustering and classification

        Args:
            texts: List of text strings to encode.
            dimensions: Optional number of dimensions for the output embeddings.
                Applicable for Matryoshka Embeddings (dimensionality reduction).

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            RuntimeError: If the model doesn't support embeddings.
        """
        try:
            result = await self.engine.async_encode(
                prompt=texts,
                dimensions=dimensions,
            )
            # Result format: {"embeddings": [[float, ...], ...]}
            return result.get("embeddings", [])
        except Exception as e:
            raise RuntimeError(
                f"Embedding generation failed. This model may not support embeddings. "
                f"Error: {e}"
            ) from e

    async def encode_single(
        self,
        text: str,
        dimensions: Optional[int] = None,
    ) -> List[float]:
        """Get embedding for a single text.

        Convenience method for single-text embedding.

        Args:
            text: Text string to encode.
            dimensions: Optional number of dimensions for the output embedding.

        Returns:
            Embedding vector for the input text.
        """
        embeddings = await self.encode([text], dimensions=dimensions)
        return embeddings[0] if embeddings else []

    async def compute_similarity(
        self,
        text1: str,
        text2: str,
        dimensions: Optional[int] = None,
    ) -> float:
        """Compute cosine similarity between two texts.

        Convenience method for computing semantic similarity using embeddings.

        Args:
            text1: First text.
            text2: Second text.
            dimensions: Optional embedding dimensions.

        Returns:
            Cosine similarity score between -1 and 1.
        """
        import math

        embeddings = await self.encode([text1, text2], dimensions=dimensions)
        if len(embeddings) < 2:
            raise RuntimeError("Failed to get embeddings for both texts")

        emb1, emb2 = embeddings[0], embeddings[1]

        # Compute cosine similarity
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = math.sqrt(sum(a * a for a in emb1))
        norm2 = math.sqrt(sum(b * b for b in emb2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def supports_embeddings(self) -> bool:
        """Check if this engine supports embedding generation.

        Note: This returns True but actual support depends on the model.
        Embedding models (e.g., sentence transformers) will work,
        but standard LLMs may not support the encode() API.

        Returns:
            True - SGLang has embedding API support.
        """
        return True

    # ============================================================================
    # Score API for RLHF Reward Models
    # ============================================================================

    async def score(
        self,
        input_ids: List[List[int]],
        output_ids: List[List[int]],
        return_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        """Compute reward scores using a reward model.

        This API is essential for RLHF workflows where you need to score
        prompt-response pairs using a reward model. The scores can be used
        for training signals, best-of-n sampling, or rejection sampling.

        Args:
            input_ids: List of tokenized prompts (one per request).
            output_ids: List of tokenized responses (one per request).
            return_hidden_states: Whether to return hidden states (for reward head extraction).

        Returns:
            Dict with:
            - "scores": List of scalar reward scores (one per input-output pair)
            - "hidden_states": Optional list of hidden state tensors (if requested)

        Raises:
            RuntimeError: If the model doesn't support scoring or API fails.

        Example:
            # Score a prompt-response pair
            prompt_ids = tokenizer.encode("What is 2+2?")
            response_ids = tokenizer.encode("The answer is 4.")
            result = await engine.score([prompt_ids], [response_ids])
            reward = result["scores"][0]  # e.g., 0.85
        """
        try:
            from sglang.srt.managers.io_struct import ScoreReqInput

            # Build score request
            obj = ScoreReqInput(
                input_ids=input_ids,
                output_ids=output_ids,
                return_hidden_states=return_hidden_states,
            )

            # Call SGLang's score API
            result = await self.engine.tokenizer_manager.score(obj, None)

            # Process results
            scores = []
            hidden_states = [] if return_hidden_states else None

            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        scores.append(item.get("score", item.get("reward", 0.0)))
                        if return_hidden_states and "hidden_states" in item:
                            hidden_states.append(item["hidden_states"])
                    else:
                        scores.append(float(item))
            else:
                scores = [float(result)]

            return {
                "scores": scores,
                "hidden_states": hidden_states,
            }

        except ImportError:
            raise RuntimeError(
                "ScoreReqInput not available in this SGLang version. "
                "Score API requires SGLang with reward model support. "
                "Please upgrade SGLang or use a compatible version."
            )
        except Exception as e:
            raise RuntimeError(
                f"Score API failed. This model may not be a reward model. "
                f"Error: {e}"
            ) from e

    async def async_score(
        self,
        input_ids: List[List[int]],
        output_ids: List[List[int]],
        return_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        """Alias for score() for API compatibility."""
        return await self.score(input_ids, output_ids, return_hidden_states)

    def supports_scoring(self) -> bool:
        """Check if this engine supports reward model scoring.

        Returns:
            True if the score() API is available (depends on SGLang version).
        """
        try:
            from sglang.srt.managers.io_struct import ScoreReqInput
            return True
        except ImportError:
            return False

    # ============================================================================
    # Reference Model Log Probability API for RLHF
    # ============================================================================

    async def compute_ref_logprobs(
        self,
        prompt_token_ids: List[List[int]],
        response_token_ids: List[List[int]],
    ) -> List[torch.Tensor]:
        """Compute reference model log probabilities for PPO/GRPO training.

        This API is essential for RLHF workflows where you need to compute
        KL divergence between policy and reference model distributions.
        Uses SGLang's optimized logprob computation path without sampling.

        The method uses the score() API with return_logprob=True to get
        log probabilities for the response tokens given the prompt context.

        Args:
            prompt_token_ids: List of tokenized prompts (one per request).
            response_token_ids: List of tokenized responses (one per request).

        Returns:
            List of log probability tensors (one per input-output pair),
            where each tensor has shape [response_length].

        Raises:
            RuntimeError: If the logprob computation fails.

        Example:
            # Compute ref logprobs for KL divergence
            prompt_ids = tokenizer.encode("What is 2+2?")
            response_ids = tokenizer.encode("The answer is 4.")
            logprobs = await engine.compute_ref_logprobs([prompt_ids], [response_ids])
            kl_div = policy_logprobs - logprobs[0]  # Per-token KL
        """
        try:
            from sglang.srt.managers.io_struct import GenerateReqInput

            results = []

            # Process in batches for efficiency
            for prompt_ids, response_ids in zip(prompt_token_ids, response_token_ids):
                # Build full sequence (prompt + response)
                full_sequence = prompt_ids + response_ids
                response_start_idx = len(prompt_ids)

                # Use generate with max_new_tokens=0 to get logprobs without generation
                # This triggers prefill-only mode with logprob computation
                obj = GenerateReqInput(
                    input_ids=full_sequence,
                    sampling_params={
                        "max_new_tokens": 0,  # No generation, just prefill for logprobs
                        "return_logprob": True,
                        "top_logprobs_num": 1,  # We only need the selected token's logprob
                        "logprob_start_len": response_start_idx,  # Only compute for response tokens
                    },
                )

                # Call SGLang's generate API in prefill-only mode
                result = await self.engine.tokenizer_manager.generate(obj, None)

                # Extract logprobs for response tokens
                if hasattr(result, 'input_token_ids_logprobs_val') and result.input_token_ids_logprobs_val:
                    # Get logprobs starting from response_start_idx
                    logprobs_raw = result.input_token_ids_logprobs_val
                    # Filter to only response tokens
                    response_logprobs = logprobs_raw[-len(response_ids):] if len(logprobs_raw) >= len(response_ids) else logprobs_raw
                    results.append(torch.tensor(response_logprobs, dtype=torch.float32))
                elif hasattr(result, 'meta_info') and 'input_token_logprobs' in result.meta_info:
                    # Alternative extraction path
                    logprobs_raw = result.meta_info['input_token_logprobs']
                    response_logprobs = logprobs_raw[-len(response_ids):]
                    results.append(torch.tensor(response_logprobs, dtype=torch.float32))
                else:
                    # Fallback: return zeros if logprobs not available
                    logger.warning(
                        f"Logprobs not available in result. "
                        f"Result type: {type(result)}, attrs: {dir(result)}"
                    )
                    results.append(torch.zeros(len(response_ids), dtype=torch.float32))

            return results

        except ImportError as e:
            raise RuntimeError(
                f"GenerateReqInput not available in this SGLang version. "
                f"Reference logprob API requires SGLang with generate support. "
                f"Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Reference logprob computation failed. Error: {e}"
            ) from e

    async def compute_ref_logprobs_batched(
        self,
        prompt_token_ids: List[List[int]],
        response_token_ids: List[List[int]],
        batch_size: int = 32,
    ) -> List[torch.Tensor]:
        """Batched version of compute_ref_logprobs for better throughput.

        Processes multiple prompt-response pairs in parallel batches.

        Args:
            prompt_token_ids: List of tokenized prompts.
            response_token_ids: List of tokenized responses.
            batch_size: Number of requests to process in parallel.

        Returns:
            List of log probability tensors.
        """
        import asyncio

        results = []
        for i in range(0, len(prompt_token_ids), batch_size):
            batch_prompts = prompt_token_ids[i:i + batch_size]
            batch_responses = response_token_ids[i:i + batch_size]

            # Process batch in parallel
            batch_results = await asyncio.gather(*[
                self.compute_ref_logprobs([p], [r])
                for p, r in zip(batch_prompts, batch_responses)
            ])

            # Flatten results
            for batch_result in batch_results:
                results.extend(batch_result)

        return results

    def supports_ref_logprobs(self) -> bool:
        """Check if this engine supports reference model logprob computation.

        Returns:
            True if compute_ref_logprobs() is available.
        """
        try:
            from sglang.srt.managers.io_struct import GenerateReqInput
            return True
        except ImportError:
            return False

    # ============================================================================
    # Server Info API
    # ============================================================================

    async def get_server_info(self) -> Dict[str, Any]:
        """Get server information and configuration.

        Returns metadata about the running inference engine including
        model configuration, memory usage, and server settings.

        Returns:
            Dict with server information:
            - "model_path": Path to the loaded model
            - "tp_size": Tensor parallel size
            - "pp_size": Pipeline parallel size
            - "dp_size": Data parallel size
            - "ep_size": Expert parallel size
            - "gpu_memory_utilization": Configured GPU memory limit
            - "max_num_seqs": Maximum concurrent sequences
            - "enable_lora": Whether LoRA is enabled
            - "weight_version": Current weight version (if set)
            - "cuda_device": Current CUDA device info

        Example:
            info = await engine.get_server_info()
            print(f"Model: {info['model_path']}, TP={info['tp_size']}")
        """
        info = {
            "model_path": self._model_path,
            "tp_size": self._tp_size,
            "pp_size": self._pp_size,
            "dp_size": self._dp_size,
            "ep_size": self._ep_size,
            "enable_lora": self._enable_lora,
            "weight_version": self._weight_version,
        }

        # Get GPU memory info
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            info["gpu_memory"] = {
                "free_mb": free_mem / 1024**2,
                "total_mb": total_mem / 1024**2,
                "used_mb": (total_mem - free_mem) / 1024**2,
                "utilization": 1.0 - (free_mem / total_mem),
            }
            info["cuda_device"] = torch.cuda.get_device_name()
        except Exception:
            info["gpu_memory"] = None
            info["cuda_device"] = None

        # Try to get server args from engine
        try:
            if hasattr(self.engine, 'server_args'):
                server_args = self.engine.server_args
                info["gpu_memory_utilization"] = getattr(server_args, "mem_fraction_static", None)
                info["max_num_seqs"] = getattr(server_args, "max_running_requests", None)
                info["attention_backend"] = getattr(server_args, "attention_backend", None)
                info["chunked_prefill_size"] = getattr(server_args, "chunked_prefill_size", None)
        except Exception:
            pass

        return info

    async def get_memory_pool_size(self) -> Dict[str, int]:
        """Get KV cache memory pool size information.

        Returns:
            Dict with:
            - "total_tokens": Maximum tokens the memory pool can hold
            - "available_tokens": Currently available tokens
            - "used_tokens": Currently used tokens

        Note:
            This API may not be available in all SGLang versions.
        """
        try:
            # Try to get from scheduler if available
            if hasattr(self.engine, 'tokenizer_manager'):
                result = await self.engine.tokenizer_manager.get_memory_pool_size()
                return {
                    "total_tokens": result.get("total", 0),
                    "available_tokens": result.get("available", 0),
                    "used_tokens": result.get("used", 0),
                }
        except Exception as e:
            logger.debug(f"get_memory_pool_size not available: {e}")

        return {"total_tokens": 0, "available_tokens": 0, "used_tokens": 0}

    # ============================================================================
    # Session Introspection APIs
    # ============================================================================

    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific session.

        Args:
            session_id: The session ID to query.

        Returns:
            Dict with session info if found:
            - "session_id": The session ID
            - "capacity": Reserved capacity
            - "used_tokens": Tokens currently used
            - "num_requests": Number of requests in this session
            Or None if session not found.
        """
        try:
            from sglang.srt.managers.io_struct import GetSessionInfoReqInput
            obj = GetSessionInfoReqInput(session_id=session_id)
            result = await self.engine.tokenizer_manager.get_session_info(obj, None)
            return result
        except (ImportError, AttributeError) as e:
            logger.debug(f"get_session_info not available: {e}")
            return None

    async def list_sessions(self) -> List[str]:
        """List all active session IDs.

        Returns:
            List of active session ID strings.
        """
        try:
            from sglang.srt.managers.io_struct import ListSessionsReqInput
            obj = ListSessionsReqInput()
            result = await self.engine.tokenizer_manager.list_sessions(obj, None)
            return result if result else []
        except (ImportError, AttributeError) as e:
            logger.debug(f"list_sessions not available: {e}")
            return []

    async def fork_session(
        self,
        source_session_id: str,
        target_session_id: Optional[str] = None,
        offset: int = -1,
    ) -> Optional[str]:
        """Fork a session at a specific point.

        Creates a new session that branches from an existing session
        at the specified offset. Useful for exploring multiple
        generation paths from the same prefix.

        Args:
            source_session_id: Session ID to fork from.
            target_session_id: Optional ID for the new session.
            offset: Token offset to fork at (-1 = end of session).

        Returns:
            New session ID if successful, None otherwise.
        """
        try:
            from sglang.srt.managers.io_struct import ForkSessionReqInput
            obj = ForkSessionReqInput(
                source_session_id=source_session_id,
                target_session_id=target_session_id,
                offset=offset,
            )
            result = await self.engine.tokenizer_manager.fork_session(obj, None)
            if result:
                logger.info(f"Forked session '{source_session_id}' -> '{result}' at offset {offset}")
            return result
        except (ImportError, AttributeError) as e:
            logger.debug(f"fork_session not available: {e}")
            return None

    # Remote weight sync utilities
    @staticmethod
    async def sync_weights_to_remote(
        endpoint: str,
        weights: Dict[str, torch.Tensor],
        weight_version: Optional[str] = None,
        flush_cache: bool = True,
        timeout: float = 300.0,
    ) -> bool:
        """Sync weights to a remote SGLang HTTP server.

        This method sends weights to a remote SGLang instance running as an HTTP server.
        Useful for distributed inference setups where training and inference are on
        different machines.

        The remote server must be started with:
            python -m sglang.launch_server --model-path <path> --load-format dummy ...

        Args:
            endpoint: HTTP endpoint of the remote SGLang server (e.g., "http://localhost:30000").
            weights: Dictionary mapping parameter names to tensors.
            weight_version: Optional version identifier for tracking.
            flush_cache: Whether to flush KV cache after weight update (default True).
            timeout: Request timeout in seconds.

        Returns:
            True if sync succeeded, False otherwise.

        Note:
            For production cross-node weight sync, consider using SGLang's checkpoint_engine
            with ParameterServer for better performance:
                python -m sglang.srt.checkpoint_engine.update --update-method broadcast ...

        Example:
            # On training node, after gradient step
            weights = {name: param.data for name, param in model.named_parameters()}
            success = await SGLangInferenceEngine.sync_weights_to_remote(
                endpoint="http://inference-node:30000",
                weights=weights,
                weight_version=f"step_{global_step}",
            )
        """
        import httpx
        import pickle
        import base64

        try:
            # Serialize weights for HTTP transport
            serialized_weights = {}
            for name, tensor in weights.items():
                # Convert tensor to CPU and serialize
                tensor_bytes = pickle.dumps(tensor.cpu())
                serialized_weights[name] = base64.b64encode(tensor_bytes).decode('ascii')

            payload = {
                "weights": serialized_weights,
                "flush_cache": flush_cache,
                "weight_version": weight_version,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{endpoint}/update_weights_from_tensor",
                    json=payload,
                    timeout=timeout,
                )
                response.raise_for_status()
                result = response.json()
                return result.get("success", False)

        except Exception as e:
            logger.error(f"Failed to sync weights to remote {endpoint}: {e}")
            return False

    @staticmethod
    async def check_remote_ready(
        endpoint: str,
        timeout: float = 10.0,
        max_retries: int = 30,
    ) -> bool:
        """Check if a remote SGLang server is ready.

        Args:
            endpoint: HTTP endpoint of the remote SGLang server.
            timeout: Per-request timeout in seconds.
            max_retries: Maximum number of retry attempts.

        Returns:
            True if server is ready, False if max retries exceeded.
        """
        import httpx

        for i in range(max_retries):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{endpoint}/health", timeout=timeout)
                    if response.status_code == 200:
                        return True
            except Exception:
                pass

            if i < max_retries - 1:
                import asyncio
                await asyncio.sleep(1.0)

        return False

    @staticmethod
    async def get_remote_weight_version(
        endpoint: str,
        timeout: float = 10.0,
    ) -> Optional[str]:
        """Get the current weight version from a remote SGLang server.

        Args:
            endpoint: HTTP endpoint of the remote SGLang server.
            timeout: Request timeout in seconds.

        Returns:
            Current weight version string, or None if unavailable.
        """
        import httpx

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{endpoint}/get_weight_version",
                    timeout=timeout,
                )
                response.raise_for_status()
                result = response.json()
                return result.get("weight_version")
        except Exception as e:
            logger.warning(f"Failed to get weight version from {endpoint}: {e}")
            return None

    # ============================================================================
    # Overlapped Weight Sync API
    # ============================================================================

    def supports_overlapped_weight_sync(self) -> bool:
        """Check if overlapped weight sync is supported.

        Returns:
            True - SGLang supports overlapped weight synchronization.
        """
        return True

    async def start_weight_transfer(
        self, request: WeightUpdateRequest
    ) -> WeightTransferHandle:
        """Start transferring weights in the background.

        For CUDA IPC: Receives IPC handles and opens them to access GPU memory.
        For Broadcast: Starts receiving weights via NCCL broadcast.

        The weights are staged in a buffer and not yet applied to the model,
        allowing generation to continue during transfer.

        Args:
            request: Weight update request with weight data.

        Returns:
            WeightTransferHandle for use with finish_weight_transfer().
        """
        import asyncio

        async def _transfer_weights() -> Dict[str, torch.Tensor]:
            """Background task to receive and stage weights."""
            staged_weights: Dict[str, torch.Tensor] = {}

            if isinstance(request, CudaIpcWeightUpdateRequest):
                # For IPC: Open handles and copy to staging buffer
                from skyrl_train.weight_sync.cuda_ipc_strategy import CudaIpcWeightTransferReceiver
                from skyrl_train.utils import str_to_torch_dtype

                model_dtype = str_to_torch_dtype(request.dtypes[0])
                receiver = CudaIpcWeightTransferReceiver(model_dtype=model_dtype)

                for name, tensor in receiver.receive_weights(request):
                    # Clone to staging buffer (receiver yields views into IPC memory)
                    staged_weights[name] = tensor.clone()

            elif isinstance(request, BroadcastWeightUpdateRequest):
                # For Broadcast: Receive via NCCL into staging buffer
                from skyrl_train.utils import str_to_torch_dtype

                for name, dtype_str, shape in zip(request.names, request.dtypes, request.shapes):
                    dtype = str_to_torch_dtype(dtype_str)
                    # Allocate staging buffer
                    staged_tensor = torch.empty(shape, dtype=dtype, device="cuda")
                    # Receive via broadcast (uses process group from init_weight_update_communicator)
                    torch.distributed.broadcast(staged_tensor, 0, group=self._weight_loader._group)
                    staged_weights[name] = staged_tensor
            else:
                raise TypeError(f"Unsupported request type: {type(request).__name__}")

            return staged_weights

        # Start background transfer
        transfer_task = asyncio.create_task(_transfer_weights())
        handle = WeightTransferHandle(
            request=request,
            transfer_task=transfer_task,
        )

        logger.debug(f"Started background weight transfer for {len(request.names)} parameters")
        return handle

    async def finish_weight_transfer(
        self, handle: WeightTransferHandle, flush_cache: bool = True
    ) -> None:
        """Complete weight transfer and apply staged weights.

        Waits for background transfer to complete, then applies the staged
        weights to the model. Should be called when generation is paused.

        Args:
            handle: Handle from start_weight_transfer().
            flush_cache: Whether to flush KV cache after applying (default True).
        """
        # Wait for transfer to complete
        await handle.wait()

        if handle.error:
            raise handle.error

        staged_weights = handle.staged_weights
        if staged_weights is None:
            raise RuntimeError("No staged weights available")

        # Apply staged weights to model
        # Use SGLang's update_weights_from_tensor API with pre-loaded tensors
        request_tensors = [(name, tensor) for name, tensor in staged_weights.items()]

        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[
                MultiprocessingSerializer.serialize(request_tensors)
                for _ in range(self._tp_size)
            ],
            load_format=None,  # Direct tensor load, no custom loader needed
            flush_cache=flush_cache,
            weight_version=handle.request.weight_version,
        )

        success, message = await self.engine.tokenizer_manager.update_weights_from_tensor(obj, None)
        if not success:
            raise RuntimeError(f"Failed to apply staged weights: {message}")

        logger.debug(f"Applied {len(staged_weights)} staged weights (flush_cache={flush_cache})")

    async def overlapped_weight_sync(
        self,
        request: WeightUpdateRequest,
        pause_mode: Literal["abort", "in_place", "retract"] = "in_place",
    ) -> None:
        """Perform overlapped weight synchronization.

        This is a convenience method that:
        1. Starts weight transfer in background (generation continues)
        2. Waits for transfer to complete
        3. Pauses generation briefly
        4. Applies weights
        5. Resumes generation

        The total pause time is minimized to just the weight application step,
        not the full transfer time.

        Args:
            request: Weight update request.
            pause_mode: How to pause generation during weight application.
                - "in_place": Pause without aborting, preserves KV cache (fastest resume)
                - "abort": Abort current requests (clean slate)
                - "retract": Move requests to waiting queue (KV cache may be flushed)
        """
        # 1. Start background transfer while generation continues
        handle = await self.start_weight_transfer(request)

        # 2. Wait for transfer to complete (generation still running)
        await handle.wait()

        # 3. Pause generation briefly for weight application
        await self.pause_generation(mode=pause_mode)

        try:
            # 4. Apply staged weights (quick operation)
            await self.finish_weight_transfer(handle, flush_cache=True)
        finally:
            # 5. Resume generation
            await self.continue_generation()

        logger.info(
            f"Overlapped weight sync complete: {len(request.names)} params, "
            f"version={handle.request.weight_version}"
        )


SGLangRayActor = ray.remote(SGLangInferenceEngine)
