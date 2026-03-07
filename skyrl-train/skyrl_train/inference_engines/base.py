from abc import ABC, abstractmethod
from typing import List, Dict, TypedDict, Any, Optional, Hashable, Literal, AsyncIterator, TYPE_CHECKING

if TYPE_CHECKING:
    from skyrl_train.weight_sync.transfer_strategy import WeightSyncInitInfo
    from skyrl_train.weight_sync import WeightUpdateRequest

MessageType = Dict[str, str]
ConversationType = List[MessageType]


class InferenceEngineInput(TypedDict):
    # Either prompts or prompt_token_ids must be provided, but not both.
    prompts: Optional[List[ConversationType]]
    prompt_token_ids: Optional[List[List[int]]]
    sampling_params: Optional[Dict[str, Any]]
    session_ids: Optional[List[Hashable]]
    # Whether to return hidden states from the model (for value function, representation learning)
    return_hidden_states: Optional[bool]
    # Capture routed experts for R3 rollout
    return_routed_experts: Optional[bool]
    # Multimodal support: image data for vision-language models.
    # Can be a list of image inputs (one per prompt), where each can be:
    # - str: File path, URL, or base64-encoded image
    # - bytes: Raw image bytes
    # - PIL.Image.Image: PIL Image object
    # - List[...]: Multiple images per prompt
    # - None: No image for this prompt
    image_data: Optional[List[Any]]
    # Multimodal support: video data for video-language models (Qwen3-VL, NVILA, LLaVA-NeXT-Video, etc.)
    # Same format as image_data: file paths, URLs, base64, or bytes.
    video_data: Optional[List[Any]]
    # Multimodal support: audio data for audio-language models (Qwen-Audio, MiniCPM-o, Phi-4-multimodal, etc.)
    # Same format as image_data: file paths, URLs, base64, or bytes.
    audio_data: Optional[List[Any]]


class InferenceEngineOutput(TypedDict):
    # We always return both tokens and text outputs. The tokens are the outputs
    # of inference engine, and the text is the decoded text output. Therefore,
    # it is guaranteed that tokenizer.decode(response_token_ids, skip_special_tokens=True) == responses,
    # but the reverse is not guaranteed, since there are multiple ways to
    # represent the same text with tokens. Therefore, for multi-turn generation,
    # please use token-in-token-out to ensure correctness.
    # `skip_special_tokens=True` is needed because string responses do not include EOS tokens like `<|im_end|>`
    responses: List[str]
    response_ids: List[List[int]]
    stop_reasons: List[str]
    response_logprobs: Optional[List[List[float]]]
    # Weight version identifier for tracking which training step's weights generated this output.
    # Useful for correlating inference samples with training steps in RL workflows.
    weight_version: Optional[str]
    # Number of samples generated per prompt (for n>1 sampling).
    # When n_per_prompt > 1, outputs are flattened but grouped by prompt:
    # [prompt0_sample0, prompt0_sample1, ..., prompt0_sampleN-1, prompt1_sample0, ...]
    # To reconstruct groups: outputs[i*n:(i+1)*n] gives all samples for prompt i.
    # Default is 1 (single sample per prompt).
    n_per_prompt: Optional[int]
    # Request IDs for session-based generation (SGLang only).
    # Used to continue multi-turn conversations within a session.
    # Pass the request_id to subsequent generate_with_session() calls as the 'rid' parameter.
    request_ids: Optional[List[str]]
    # Hidden states from the final layer (SGLang only, requires enable_return_hidden_states=True).
    # Useful for value function enrichment, representation learning, and RL state extraction.
    # Each entry corresponds to a generated response.
    hidden_states: Optional[List[Any]]
    # numpy int32 arrays of shape(response_len * num_layers * top_k)
    routed_experts: Optional[List[Any]]


class StreamingChunk(TypedDict):
    """A single chunk from streaming generation.

    Yields incremental output as tokens are generated, enabling real-time
    processing and early stopping based on partial outputs.
    """
    # Index of the request in the batch (for batched streaming)
    index: int
    # Incremental text since last chunk (decoded token)
    delta_text: str
    # Incremental token ID since last chunk
    delta_token_id: Optional[int]
    # Log probability of the delta token (if requested)
    delta_logprob: Optional[float]
    # Whether this is the final chunk for this request
    is_finished: bool
    # Stop reason if is_finished is True
    stop_reason: Optional[str]
    # Cumulative text so far (optional, may not be provided for efficiency)
    cumulative_text: Optional[str]
    # Cumulative token IDs so far (optional, may not be provided for efficiency)
    cumulative_token_ids: Optional[List[int]]


class InferenceEngineInterface(ABC):

    @abstractmethod
    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        raise NotImplementedError

    @abstractmethod
    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handles OpenAI-compatible HTTP endpoint.

        Accepts a JSON payload: {"json": <request-body>, "headers": <headers-dict>}.
        The request body will be used to construct a ChatCompletionRequest.
        Returns a plain dict, either a ChatCompletionResponse or an ErrorResponse.
        The specific fields of the response/request depend on the engine's backend (e.g. for vllm
        these are defined in vllm.entrypoints.openai.protocol).
        """
        raise NotImplementedError

    @abstractmethod
    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handles OpenAI-compatible HTTP endpoint.

        Accepts a JSON payload: {"json": <request-body>, "headers": <headers-dict>}.
        The request body will be used to construct a CompletionRequest.
        Returns a plain dict, either a CompletionResponse or an ErrorResponse.
        The specific fields of the response/request depend on the engine's backend (e.g. for vllm
        these are defined in vllm.entrypoints.openai.protocol).
        """
        raise NotImplementedError

    @abstractmethod
    async def wake_up(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    @abstractmethod
    async def sleep(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    @abstractmethod
    async def init_weight_update_communicator(self, init_info: "WeightSyncInitInfo"):
        """Initialize weight update communicator from init info.

        Args:
            init_info: WeightSyncInitInfo from the sender containing all info needed
                to create the appropriate receiver.
        """
        raise NotImplementedError()

    @abstractmethod
    async def update_named_weights(self, request: "WeightUpdateRequest"):
        raise NotImplementedError()

    @abstractmethod
    async def teardown(self):
        raise NotImplementedError

    @abstractmethod
    async def reset_prefix_cache(self):
        raise NotImplementedError

    @abstractmethod
    def tp_size(self) -> int:
        """Return the tensor parallel size of this inference engine."""
        raise NotImplementedError

    @abstractmethod
    def pp_size(self) -> int:
        """Return the pipeline parallel size of this inference engine."""
        raise NotImplementedError

    @abstractmethod
    def dp_size(self) -> int:
        """Return the data parallel size of this inference engine."""
        raise NotImplementedError

    @abstractmethod
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
                - "retract": Pause and retract all running requests to waiting queue.
                    KV cache can be flushed and will be recomputed on continue.
        """
        raise NotImplementedError

    @abstractmethod
    async def continue_generation(self) -> None:
        """Resume generation after pause.

        Must be called after pause_generation with mode='in_place' or 'retract'
        to resume processing of paused/retracted requests.
        """
        raise NotImplementedError

    @abstractmethod
    async def abort_generation(self) -> None:
        """
        Abort all running and waiting requests, which make the ongoing requests return the
        already-generated tokens with a stop_reason of "abort". If the request was waiting,
        it returns a response with zero completion tokens.

        Convenience method that calls pause_generation with mode='abort'.
        """
        raise NotImplementedError

    async def generate_stream(
        self, input_batch: InferenceEngineInput
    ) -> AsyncIterator["StreamingChunk"]:
        """Generate responses with streaming output.

        Yields StreamingChunk objects as tokens are generated, enabling:
        - Real-time token-by-token output for interactive applications
        - Early stopping based on partial output content
        - Progress monitoring during long generations
        - Memory-efficient processing of long outputs

        Default implementation: Falls back to non-streaming generate() and yields
        the complete output as a single chunk per request.

        Args:
            input_batch: Input batch containing prompts and sampling params.

        Yields:
            StreamingChunk objects with incremental generation output.
        """
        # Default: fall back to non-streaming generate and yield complete responses
        output = await self.generate(input_batch)
        for i, (response, response_ids, stop_reason) in enumerate(
            zip(output["responses"], output["response_ids"], output["stop_reasons"])
        ):
            logprob = None
            if output.get("response_logprobs") and i < len(output["response_logprobs"]):
                logprobs = output["response_logprobs"][i]
                logprob = logprobs[-1] if logprobs else None
            yield StreamingChunk(
                index=i,
                delta_text=response,
                delta_token_id=response_ids[-1] if response_ids else None,
                delta_logprob=logprob,
                is_finished=True,
                stop_reason=stop_reason,
                cumulative_text=response,
                cumulative_token_ids=response_ids,
            )

    def supports_streaming(self) -> bool:
        """Check if this engine supports streaming generation.

        Returns:
            True if generate_stream() is available, False otherwise.
        """
        return False

    def supports_overlapped_weight_sync(self) -> bool:
        """Check if this engine supports overlapped weight synchronization.

        Overlapped weight sync allows weight transfer to happen in the background
        while generation continues, minimizing pause time to just the final
        weight application step.

        Returns:
            True if overlapped weight sync is available, False otherwise.
        """
        return False

    async def start_weight_transfer(
        self, request: "WeightUpdateRequest"
    ) -> "WeightTransferHandle":
        """Start transferring weights in the background.

        This begins the weight transfer process without blocking generation.
        The actual weight application happens when finish_weight_transfer() is called.

        Default implementation: Creates a handle that will apply weights synchronously
        when finish_weight_transfer() is called.

        Args:
            request: Weight update request with names, dtypes, shapes, and data.

        Returns:
            WeightTransferHandle that can be passed to finish_weight_transfer().
        """
        # Default: create handle with staged weights for sync application later
        return WeightTransferHandle(request=request, staged_weights=None)

    async def finish_weight_transfer(
        self, handle: "WeightTransferHandle", flush_cache: bool = True
    ) -> None:
        """Complete the weight transfer and apply weights.

        Waits for the background transfer to complete, then applies the weights.
        This should be called when generation is paused to ensure consistency.

        Default implementation: Applies weights synchronously using update_named_weights.

        Args:
            handle: Handle from start_weight_transfer().
            flush_cache: Whether to flush KV cache after applying weights.
        """
        # Default: apply weights synchronously
        await handle.wait()
        await self.update_named_weights(handle.request)
        if flush_cache:
            await self.reset_prefix_cache()

    # Score API for RLHF reward models
    def supports_scoring(self) -> bool:
        """Check if this engine supports reward model scoring.

        Returns:
            True if the score() API is available, False otherwise.
        """
        return False

    async def score(
        self,
        input_ids: List[List[int]],
        output_ids: List[List[int]],
        return_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        """Compute reward scores using a reward model.

        This API is essential for RLHF workflows where you need to score
        prompt-response pairs using a reward model.

        Default implementation: Returns uniform scores of 0.0 for all pairs.
        Override this in subclasses that have actual reward model capabilities.

        Args:
            input_ids: List of tokenized prompts (one per request).
            output_ids: List of tokenized responses (one per request).
            return_hidden_states: Whether to return hidden states.

        Returns:
            Dict with:
            - "scores": List of scalar reward scores
            - "hidden_states": Optional list of hidden state tensors
        """
        import warnings
        warnings.warn(
            f"{self.__class__.__name__} does not have a reward model. "
            "Returning zero scores. Override score() for actual scoring.",
            UserWarning
        )
        result: Dict[str, Any] = {"scores": [0.0] * len(input_ids)}
        if return_hidden_states:
            result["hidden_states"] = [None] * len(input_ids)
        return result

    async def get_server_info(self) -> Dict[str, Any]:
        """Get server information and configuration.

        Returns metadata about the running inference engine.

        Default implementation: Returns basic info from available methods.

        Returns:
            Dict with server information including model_path, tp_size,
            memory usage, etc.
        """
        return {
            "engine_class": self.__class__.__name__,
            "tp_size": self.tp_size(),
            "pp_size": self.pp_size(),
            "dp_size": self.dp_size(),
            "supports_streaming": self.supports_streaming(),
            "supports_scoring": self.supports_scoring(),
            "supports_embeddings": self.supports_embeddings(),
            "supports_sessions": self.supports_sessions(),
            "supports_overlapped_weight_sync": self.supports_overlapped_weight_sync(),
        }

    async def get_memory_pool_size(self) -> Dict[str, int]:
        """Get KV cache memory pool size information.

        Default implementation: Returns unknown/placeholder values.
        Override in subclasses with actual memory tracking.

        Returns:
            Dict with total_tokens, available_tokens, used_tokens.
        """
        return {
            "total_tokens": -1,  # Unknown
            "available_tokens": -1,  # Unknown
            "used_tokens": -1,  # Unknown
        }

    # Model saving API
    async def save_sharded_model(
        self,
        save_path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> bool:
        """Save model weights to disk in sharded format.

        Default implementation: Logs a warning and returns False.
        Override in subclasses with actual model saving capabilities.

        Args:
            save_path: Directory path to save the model.
            pattern: Optional pattern for shard naming.
            max_size: Maximum size per shard in bytes.

        Returns:
            True if successful, False otherwise.
        """
        import warnings
        warnings.warn(
            f"{self.__class__.__name__} does not support save_sharded_model. "
            "Override this method in your inference engine implementation.",
            UserWarning
        )
        return False

    async def save_remote_model(
        self,
        remote_path: str,
        storage_type: str = "s3",
    ) -> bool:
        """Save model weights to remote storage.

        Default implementation: Falls back to save_sharded_model to a temp directory,
        then uploads to remote storage.

        Args:
            remote_path: Remote storage path.
            storage_type: Storage backend type (s3, gcs, azure).

        Returns:
            True if successful, False otherwise.
        """
        import tempfile
        import os
        import shutil

        # Save to temp directory first
        with tempfile.TemporaryDirectory() as temp_dir:
            local_success = await self.save_sharded_model(temp_dir)
            if not local_success:
                return False

            # Upload based on storage type
            try:
                if storage_type == "s3":
                    import boto3
                    s3 = boto3.client("s3")
                    # Parse s3://bucket/key format
                    if remote_path.startswith("s3://"):
                        remote_path = remote_path[5:]
                    parts = remote_path.split("/", 1)
                    bucket = parts[0]
                    prefix = parts[1] if len(parts) > 1 else ""

                    for root, _, files in os.walk(temp_dir):
                        for file in files:
                            local_file = os.path.join(root, file)
                            rel_path = os.path.relpath(local_file, temp_dir)
                            s3_key = os.path.join(prefix, rel_path) if prefix else rel_path
                            s3.upload_file(local_file, bucket, s3_key)
                    return True

                elif storage_type == "gcs":
                    from google.cloud import storage
                    client = storage.Client()
                    # Parse gs://bucket/prefix format
                    if remote_path.startswith("gs://"):
                        remote_path = remote_path[5:]
                    parts = remote_path.split("/", 1)
                    bucket_name = parts[0]
                    prefix = parts[1] if len(parts) > 1 else ""
                    bucket = client.bucket(bucket_name)

                    for root, _, files in os.walk(temp_dir):
                        for file in files:
                            local_file = os.path.join(root, file)
                            rel_path = os.path.relpath(local_file, temp_dir)
                            blob_name = os.path.join(prefix, rel_path) if prefix else rel_path
                            blob = bucket.blob(blob_name)
                            blob.upload_from_filename(local_file)
                    return True

                elif storage_type == "azure":
                    from azure.storage.blob import BlobServiceClient
                    from azure.identity import DefaultAzureCredential
                    # Parse azure://container/prefix or https://account.blob.core.windows.net/container/prefix
                    credential = DefaultAzureCredential()
                    if "blob.core.windows.net" in remote_path:
                        # Full URL format
                        blob_service = BlobServiceClient(account_url=remote_path.rsplit("/", 2)[0], credential=credential)
                        parts = remote_path.split("/")
                        container_name = parts[-2] if len(parts) >= 2 else parts[-1]
                        prefix = parts[-1] if len(parts) >= 2 else ""
                    else:
                        import warnings
                        warnings.warn(f"Azure path format not recognized: {remote_path}", UserWarning)
                        return False

                    container = blob_service.get_container_client(container_name)
                    for root, _, files in os.walk(temp_dir):
                        for file in files:
                            local_file = os.path.join(root, file)
                            rel_path = os.path.relpath(local_file, temp_dir)
                            blob_name = os.path.join(prefix, rel_path) if prefix else rel_path
                            with open(local_file, "rb") as data:
                                container.upload_blob(name=blob_name, data=data, overwrite=True)
                    return True

                else:
                    import warnings
                    warnings.warn(f"Unsupported storage type: {storage_type}. Supported: s3, gcs, azure", UserWarning)
                    return False

            except Exception as e:
                import warnings
                warnings.warn(f"Failed to upload to {storage_type}: {e}", UserWarning)
                return False

    # Decode API
    async def decode(
        self,
        token_ids: List[List[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Decode token IDs to text strings.

        Default implementation: Uses a basic fallback that attempts to load
        a tokenizer. Override in subclasses with actual tokenizer access.

        Args:
            token_ids: List of token ID sequences to decode.
            skip_special_tokens: Whether to skip special tokens.

        Returns:
            List of decoded text strings.
        """
        import warnings
        warnings.warn(
            f"{self.__class__.__name__} does not have direct tokenizer access. "
            "Returning placeholder strings. Override decode() for actual decoding.",
            UserWarning
        )
        # Return placeholder - subclasses should override with actual tokenizer
        return [f"<decoded:{len(ids)} tokens>" for ids in token_ids]

    # Profiling API
    _profiling_active: bool = False
    _profile_start_time: Optional[float] = None
    _profile_data: Optional[Dict[str, Any]] = None

    async def start_profile(self) -> bool:
        """Start profiling the inference engine.

        Default implementation: Tracks basic timing information.
        Override in subclasses for detailed GPU/memory profiling.

        Returns:
            True if profiling started successfully.
        """
        import time
        self._profiling_active = True
        self._profile_start_time = time.time()
        self._profile_data = {
            "start_time": self._profile_start_time,
            "events": [],
        }
        return True

    async def stop_profile(self) -> Optional[Dict[str, Any]]:
        """Stop profiling and collect results.

        Default implementation: Returns basic timing information.
        Override in subclasses for detailed profiling data.

        Returns:
            Dict with profiling results, or None if not available.
        """
        import time
        if not self._profiling_active or self._profile_start_time is None:
            return None

        end_time = time.time()
        self._profiling_active = False

        result = {
            "start_time": self._profile_start_time,
            "end_time": end_time,
            "duration_seconds": end_time - self._profile_start_time,
            "events": self._profile_data.get("events", []) if self._profile_data else [],
        }

        self._profile_start_time = None
        self._profile_data = None
        return result

    # Embedding API
    def supports_embeddings(self) -> bool:
        """Check if this engine supports embedding generation.

        Returns:
            True if the encode() API is available, False otherwise.
        """
        return False

    async def encode(
        self,
        texts: List[str],
        dimensions: Optional[int] = None,
    ) -> List[List[float]]:
        """Get embeddings for a list of texts.

        Default implementation: Uses a simple hash-based embedding for fallback.
        Override in subclasses with actual embedding model capabilities.

        Args:
            texts: List of text strings to encode.
            dimensions: Optional number of dimensions for output embeddings.

        Returns:
            List of embedding vectors, one per input text.
        """
        import warnings
        import hashlib
        import math

        warnings.warn(
            f"{self.__class__.__name__} does not have an embedding model. "
            "Returning hash-based pseudo-embeddings. Override encode() for actual embeddings.",
            UserWarning
        )

        dim = dimensions or 768  # Default embedding dimension
        embeddings = []

        for text in texts:
            # Create deterministic pseudo-embedding from text hash
            text_bytes = text.encode("utf-8")
            hash_bytes = hashlib.sha256(text_bytes).digest()

            # Expand hash to desired dimensions
            embedding = []
            for i in range(dim):
                byte_idx = i % len(hash_bytes)
                # Convert to float in [-1, 1] range
                val = (hash_bytes[byte_idx] / 255.0) * 2 - 1
                # Add some variation based on position
                val = math.sin(val * (i + 1))
                embedding.append(val)

            # Normalize to unit length
            norm = math.sqrt(sum(x * x for x in embedding))
            if norm > 0:
                embedding = [x / norm for x in embedding]

            embeddings.append(embedding)

        return embeddings

    async def encode_single(
        self,
        text: str,
        dimensions: Optional[int] = None,
    ) -> List[float]:
        """Get embedding for a single text.

        Args:
            text: Text string to encode.
            dimensions: Optional number of dimensions.

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
        """Compute cosine similarity between two texts using embeddings.

        This is a convenience method that encodes both texts and computes
        their cosine similarity. Useful for semantic similarity scoring
        in RLHF and reward modeling.

        Args:
            text1: First text to compare.
            text2: Second text to compare.
            dimensions: Optional embedding dimensions.

        Returns:
            Cosine similarity score between -1 and 1.
            Returns 0.0 if embeddings are not supported.
        """
        if not self.supports_embeddings():
            return 0.0
        embeddings = await self.encode([text1, text2], dimensions=dimensions)
        if len(embeddings) < 2:
            return 0.0
        # Cosine similarity
        import math
        vec1, vec2 = embeddings[0], embeddings[1]
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    # Session management for multi-turn conversations
    def supports_sessions(self) -> bool:
        """Check if this engine supports session-based generation.

        Sessions enable efficient multi-turn conversations by maintaining
        KV cache state across turns, avoiding redundant prefix recomputation.

        Returns:
            True if session APIs (open_session, generate_with_session, etc.) are available.
        """
        return False

    # Session storage for default implementation
    _sessions: Optional[Dict[str, Dict[str, Any]]] = None

    async def open_session(
        self,
        capacity_of_str_len: int = 8192,
        session_id: Optional[str] = None,
    ) -> Optional[str]:
        """Open a session for multi-turn conversation.

        Default implementation: Maintains a simple in-memory session store.
        Override in subclasses for actual KV cache-based sessions.

        Args:
            capacity_of_str_len: Maximum string/token length the session can handle.
                This reserves KV cache capacity for the session.
            session_id: Optional custom session ID. If None, auto-generated.

        Returns:
            Session ID string if successful, None if failed.
        """
        import uuid

        if self._sessions is None:
            self._sessions = {}

        if session_id is None:
            session_id = str(uuid.uuid4())

        self._sessions[session_id] = {
            "capacity": capacity_of_str_len,
            "history": [],  # List of (prompt_ids, response_ids) tuples
            "request_counter": 0,
        }
        return session_id

    async def close_session(self, session_id: str) -> None:
        """Close a session and release its resources.

        Default implementation: Removes session from in-memory store.

        Args:
            session_id: The session ID returned by open_session().
        """
        if self._sessions is not None and session_id in self._sessions:
            del self._sessions[session_id]

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

        Default implementation: Falls back to regular generate() but maintains
        session history for tracking. Override in subclasses for actual
        KV cache reuse.

        Args:
            session_id: Session ID from open_session().
            input_batch: Input batch containing prompt_token_ids and sampling_params.
            rid: Request ID to append to or branch from. For first turn, use None.
                For subsequent turns, use the request_id from the previous response.
            offset: Token offset to continue from (-1 = append at end, 0+ = branch at position).
            replace: If True, clears child branches when branching.
            drop_previous_output: If True, drops the output from previous requests.

        Returns:
            InferenceEngineOutput with generated responses. The request_ids field
            contains IDs for use in subsequent turns.
        """
        if self._sessions is None or session_id not in self._sessions:
            import warnings
            warnings.warn(f"Session {session_id} not found, creating new session", UserWarning)
            await self.open_session(session_id=session_id)

        session = self._sessions[session_id]

        # Generate using regular method
        output = await self.generate(input_batch)

        # Track history
        prompt_ids = input_batch.get("prompt_token_ids", [])
        if prompt_ids:
            for i, resp_ids in enumerate(output["response_ids"]):
                p_ids = prompt_ids[i] if i < len(prompt_ids) else []
                session["history"].append((p_ids, resp_ids))

        # Generate request IDs
        request_ids = []
        for _ in output["responses"]:
            session["request_counter"] += 1
            request_ids.append(f"{session_id}_req_{session['request_counter']}")

        # Add request_ids to output
        output["request_ids"] = request_ids
        return output


class WeightTransferHandle:
    """Handle for tracking background weight transfer progress.

    Used with start_weight_transfer() and finish_weight_transfer() for
    overlapped weight synchronization.
    """

    def __init__(
        self,
        request: "WeightUpdateRequest",
        transfer_task: Optional[Any] = None,
        staged_weights: Optional[Dict[str, Any]] = None,
    ):
        self.request = request
        self.transfer_task = transfer_task  # asyncio.Task for background transfer
        self.staged_weights = staged_weights  # Weights ready to apply
        self.is_complete = False
        self.error: Optional[Exception] = None

    async def wait(self) -> None:
        """Wait for the background transfer to complete."""
        if self.transfer_task is not None and not self.is_complete:
            try:
                self.staged_weights = await self.transfer_task
                self.is_complete = True
            except Exception as e:
                self.error = e
                raise


def group_outputs_by_prompt(output: InferenceEngineOutput) -> List[InferenceEngineOutput]:
    """Group flattened n>1 outputs into per-prompt InferenceEngineOutput objects.

    When using n>1 sampling, the output contains B*n flattened results.
    This helper function restructures them into B separate InferenceEngineOutput
    objects, each containing n samples for that prompt.

    Args:
        output: Flattened InferenceEngineOutput from n>1 sampling.

    Returns:
        List of InferenceEngineOutput objects, one per original prompt.
        Each contains n samples (responses, response_ids, etc.) for that prompt.

    Example:
        # With 2 prompts and n=3, output has 6 flattened responses
        grouped = group_outputs_by_prompt(output)
        # grouped[0] contains 3 responses for prompt 0
        # grouped[1] contains 3 responses for prompt 1
    """
    n = output.get("n_per_prompt") or 1
    if n == 1:
        return [output]

    total = len(output["responses"])
    num_prompts = total // n

    grouped: List[InferenceEngineOutput] = []
    for i in range(num_prompts):
        start = i * n
        end = start + n

        logprobs = None
        if output["response_logprobs"] is not None:
            logprobs = output["response_logprobs"][start:end]

        request_ids = None
        if output.get("request_ids") is not None:
            request_ids = output["request_ids"][start:end]

        hidden_states = None
        if output.get("hidden_states") is not None:
            hidden_states = output["hidden_states"][start:end]

        grouped.append(
            InferenceEngineOutput(
                responses=output["responses"][start:end],
                response_ids=output["response_ids"][start:end],
                stop_reasons=output["stop_reasons"][start:end],
                response_logprobs=logprobs,
                weight_version=output.get("weight_version"),
                n_per_prompt=n,
                request_ids=request_ids,
                hidden_states=hidden_states,
            )
        )

    return grouped
