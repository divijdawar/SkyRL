"""NMoE Model Wrapper for SkyRL RL Training.

This module provides NMoEModelWrapper, a wrapper class that enables nmoe models
to be used seamlessly within the SkyRL reinforcement learning training framework.

Key features:
- HFModelWrapper-compatible interface (forward, generate, gradient_checkpointing)
- NMoEModelInterface implementation for expert cache management
- Support for FP8/NVFP4 quantization with automatic cache refresh
- MoE load balancing metrics and auxiliary loss computation
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from skyrl_train.utils.torch_utils import (
        chunked_entropy_from_logits,
        logprobs_from_logits,
    )
    from skyrl_train.utils.r3_routing_replay import r3_routing_replay

    SKYRL_UTILS_AVAILABLE = True
except ImportError:
    SKYRL_UTILS_AVAILABLE = False
    # Fallback implementations for standalone usage
    def chunked_entropy_from_logits(logits, requires_grad=False, attention_mask=None):
        """Simple entropy calculation fallback."""
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        if attention_mask is not None:
            entropy = entropy * attention_mask
        return entropy

    def logprobs_from_logits(logits, labels, inplace_backward=True):
        """Simple log prob calculation fallback."""
        log_probs = torch.log_softmax(logits, dim=-1)
        gathered = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        return gathered

# Import nmoe components (optional import for flexibility)
try:
    from nmoe.model import Transformer, MoE
    from nmoe.config import Config as NMoEConfig
    from nmoe.unified import NMoEModelConfig, NMoEModelInterface
    NMOE_AVAILABLE = True
except ImportError:
    NMOE_AVAILABLE = False
    NMoEModelInterface = object  # Fallback for type hints

# Import PEFT for LoRA support (optional)
try:
    from peft import LoraConfig, TaskType, get_peft_model
    from peft.tuners.lora import LoraLayer
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def _get_nmoe_lora_target_modules(
    include_experts: bool = False,
    include_attention: bool = True,
    include_mlp: bool = False,
) -> List[str]:
    """Get default LoRA target module patterns for nmoe models.

    Args:
        include_experts: Whether to include MoE expert weights (W1, W3, W2).
        include_attention: Whether to include attention layers.
        include_mlp: Whether to include dense MLP layers (non-expert).

    Returns:
        List of module name patterns for LoRA targeting.

    Note:
        Expert weights are typically excluded from LoRA because:
        1. They are already parameter-efficient (only activated experts used)
        2. Applying LoRA to sparse experts can interfere with routing
        3. Memory savings from LoRA are less significant for experts
    """
    targets = []

    if include_attention:
        # Attention projections in nmoe
        targets.extend([
            "wq",    # Query projection
            "wk",    # Key projection
            "wv",    # Value projection
            "wo",    # Output projection
            "w_qkv", # Fused QKV (if used)
        ])

    if include_mlp:
        # Dense MLP layers (shared experts)
        targets.extend([
            "_shared.w1",
            "_shared.w3",
            "_shared.w2",
        ])

    if include_experts:
        # MoE expert weights - be careful with these
        targets.extend([
            "W1",  # Expert gate projection
            "W3",  # Expert up projection
            "W2",  # Expert down projection
        ])

    return targets


def apply_lora_to_nmoe(
    model: nn.Module,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    exclude_modules: Optional[List[str]] = None,
    init_lora_weights: str = "gaussian",
    include_experts: bool = False,
) -> nn.Module:
    """Apply LoRA adapters to an nmoe model.

    Args:
        model: The nmoe Transformer model to adapt.
        lora_rank: LoRA rank (dimension of low-rank matrices).
        lora_alpha: LoRA alpha (scaling factor).
        lora_dropout: Dropout probability for LoRA layers.
        target_modules: List of module name patterns to apply LoRA to.
            If None, uses default attention modules.
        exclude_modules: List of module name patterns to exclude from LoRA.
        init_lora_weights: Initialization method for LoRA weights.
        include_experts: Whether to include expert weights in LoRA.

    Returns:
        The model with LoRA adapters applied.

    Raises:
        ImportError: If PEFT is not installed.
        ValueError: If lora_rank <= 0.
    """
    if not PEFT_AVAILABLE:
        raise ImportError(
            "PEFT is required for LoRA support. "
            "Install it with: pip install peft"
        )

    if lora_rank <= 0:
        raise ValueError(f"lora_rank must be positive, got {lora_rank}")

    # Get target modules
    if target_modules is None:
        target_modules = _get_nmoe_lora_target_modules(
            include_experts=include_experts,
            include_attention=True,
            include_mlp=False,
        )

    # Filter out excluded modules
    if exclude_modules:
        target_modules = [m for m in target_modules if m not in exclude_modules]

    logger.info(
        f"[NMoE LoRA] Applying LoRA with rank={lora_rank}, alpha={lora_alpha}, "
        f"targets={target_modules}"
    )

    # Create LoRA config
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        init_lora_weights=init_lora_weights,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)

    # Log parameter statistics
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"[NMoE LoRA] Trainable params: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    return model


def collect_nmoe_lora_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Collect LoRA parameters from an nmoe model for saving/syncing.

    Args:
        model: The nmoe model with LoRA adapters.

    Returns:
        Dictionary mapping parameter names to tensors.
    """
    if not PEFT_AVAILABLE:
        return {}

    lora_params = {}
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            for param_name, param in module.named_parameters():
                if 'lora_' in param_name:
                    full_name = f"{name}.{param_name}" if name else param_name
                    lora_params[full_name] = param.detach().cpu()

    return lora_params


class NMoEModelWrapper(nn.Module, NMoEModelInterface):
    """Wrapper for nmoe models enabling RL training in SkyRL.

    This wrapper provides a unified interface compatible with both the SkyRL
    HFModelWrapper pattern and the NMoEModelInterface for expert cache management.

    Args:
        model_or_config: Either a pre-initialized nmoe Transformer model,
            an NMoEConfig, or NMoEModelConfig to construct the model.
        temperature: Temperature for action selection (default 1.0).
        use_torch_compile: Whether to torch.compile entropy calculation.
        gradient_checkpointing: Enable gradient checkpointing at init.

    Example:
        >>> from nmoe.config import Config
        >>> from nmoe.model import Transformer
        >>> config = Config(dim=2048, n_layers=24, n_heads=16, ...)
        >>> model = Transformer(config)
        >>> wrapper = NMoEModelWrapper(model, temperature=1.0)
        >>> log_probs = wrapper(sequences, num_actions=32, attention_mask=mask)
    """

    def __init__(
        self,
        model_or_config: Union["Transformer", "NMoEConfig", "NMoEModelConfig", Any],
        temperature: float = 1.0,
        use_torch_compile: bool = False,
        gradient_checkpointing: bool = False,
        lora_rank: int = 0,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        lora_target_modules: Optional[List[str]] = None,
        lora_exclude_modules: Optional[List[str]] = None,
        lora_init_method: str = "gaussian",
        lora_include_experts: bool = False,
        **kwargs,
    ):
        super().__init__()

        if not NMOE_AVAILABLE:
            raise ImportError(
                "nmoe package not available. Please install nmoe to use NMoEModelWrapper."
            )

        self.temperature = temperature
        self._gradient_checkpointing = False
        self._is_lora = lora_rank > 0

        # Handle different input types
        if isinstance(model_or_config, nn.Module):
            # Direct model instance
            self.model = model_or_config
        elif hasattr(model_or_config, 'dim') and hasattr(model_or_config, 'n_layers'):
            # nmoe Config or NMoEModelConfig
            self.model = Transformer(model_or_config)
            self.model.init_weights()
        else:
            raise TypeError(
                f"Expected Transformer, NMoEConfig, or NMoEModelConfig, got {type(model_or_config)}"
            )

        # Apply LoRA if requested
        if self._is_lora:
            self.model = apply_lora_to_nmoe(
                self.model,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                exclude_modules=lora_exclude_modules,
                init_lora_weights=lora_init_method,
                include_experts=lora_include_experts,
            )

        # Torch compile for entropy calculation
        self.chunked_entropy_from_logits_fn = (
            torch.compile(chunked_entropy_from_logits, dynamic=True)
            if use_torch_compile
            else chunked_entropy_from_logits
        )

        # Enable gradient checkpointing if requested
        if gradient_checkpointing:
            self.gradient_checkpointing_enable()

        # Log configuration
        self._log_model_info()

    def _log_model_info(self):
        """Log model configuration details."""
        config = self.model.config
        logger.info(f"[NMoE] Initialized wrapper for model: dim={config.dim}, "
                   f"layers={config.n_layers}, heads={config.n_heads}")
        if hasattr(config, 'n_routed_experts') and config.n_routed_experts:
            logger.info(f"[NMoE] MoE config: {config.n_routed_experts} experts, "
                       f"top-{config.n_activated_experts}, "
                       f"dtype={getattr(config, 'dtype', 'bf16')}")

    def __call__(
        self,
        input_ids_or_sequences: torch.Tensor,
        num_actions_or_attention_mask: Optional[Union[int, List[int], torch.Tensor]] = None,
        **kwargs,
    ):
        """Smart dispatch to forward() or forward_rl() based on arguments.

        This method provides backwards compatibility by automatically detecting
        whether the caller expects NMoEModelInterface.forward() or SkyRL's
        forward_rl() based on the argument types.

        - If num_actions_or_attention_mask is an int or List[int]: forward_rl()
        - Otherwise: forward() with NMoEModelInterface signature
        """
        # Detect SkyRL RL training call pattern: (sequences, num_actions, ...)
        if isinstance(num_actions_or_attention_mask, (int, list)):
            # SkyRL pattern: forward_rl(sequences, num_actions, ...)
            return self.forward_rl(
                sequences=input_ids_or_sequences,
                num_actions=num_actions_or_attention_mask,
                **kwargs,
            )
        else:
            # NMoEModelInterface pattern: forward(input_ids, attention_mask, ...)
            return self.forward(
                input_ids=input_ids_or_sequences,
                attention_mask=num_actions_or_attention_mask,
                **kwargs,
            )

    # =========================================================================
    # NMoEModelInterface.forward() Implementation
    # =========================================================================

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass returning logits and optional auxiliary outputs.

        Implements NMoEModelInterface.forward() contract.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            position_ids: Position IDs for RoPE [batch, seq_len]
            past_key_values: KV cache for incremental decoding (not supported)
            use_cache: Whether to return updated KV cache (not supported)

        Returns:
            Dict containing:
                - 'logits': Output logits [batch, seq_len, vocab_size]
                - 'past_key_values': None (KV cache not supported by nmoe)
                - 'router_logits': Router logits for aux loss (optional)
        """
        # Note: nmoe Transformer doesn't support KV cache yet
        if past_key_values is not None:
            logger.warning("[NMoE] past_key_values not supported, ignoring")

        # Forward through nmoe model
        logits = self.model(input_ids)  # [batch, seq_len, vocab_size]

        # Build output dict matching NMoEModelInterface contract
        output = {
            'logits': logits,
            'past_key_values': None,  # KV cache not supported
        }

        # Collect router logits for auxiliary loss if available
        router_logits = self._collect_router_logits()
        if router_logits is not None:
            output['router_logits'] = router_logits

        return output

    # =========================================================================
    # HFModelWrapper-Compatible Interface (SkyRL RL Training)
    # =========================================================================

    def forward_rl(
        self,
        sequences: torch.LongTensor,
        num_actions: Union[int, List[int]],
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        return_output: bool = False,
        compute_entropy: bool = False,
        entropy_requires_grad: bool = True,
        routed_experts: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Forward pass returning action log probabilities for RL training.

        Compatible with SkyRL's HFModelWrapper.forward() signature.
        Use this method for SkyRL RL training loops.

        Args:
            sequences: Input token IDs [batch, seq_len]
            num_actions: Number of action tokens at the end of sequence
            attention_mask: Attention mask [batch, seq_len] (optional)
            temperature: Sampling temperature (overrides instance default)
            return_output: If True, return (log_probs, output_dict)
            compute_entropy: If True, compute entropy in output_dict
            entropy_requires_grad: Whether entropy computation needs gradients
            routed_experts: R3 routing replay tensor [batch, response_len, num_layers, top_k] (optional)

        Returns:
            action_log_probs: Log probabilities of action tokens [batch, num_actions]
            Or (action_log_probs, output_dict) if return_output=True
        """
        # Handle position_ids from attention_mask
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        else:
            position_ids = None

        # Forward through nmoe model (with R3 routing replay if enabled)
        # nmoe Transformer takes tokens and returns logits directly
        with r3_routing_replay(self.model, routed_experts):
            logits = self.model(sequences)  # [batch, seq_len, vocab_size]

        # Apply temperature
        effective_temp = temperature if temperature != 1.0 else self.temperature
        if effective_temp != 1.0:
            logits = logits / effective_temp

        # Compute log probabilities for next token prediction
        # Roll sequences to get targets (next token for each position)
        sequences_rolled = torch.roll(sequences, shifts=-1, dims=1)

        log_probs = logprobs_from_logits(
            logits,
            sequences_rolled,
            inplace_backward=True,
        )  # [batch, seq_len]

        # Build output dict if needed
        output = {"logits": logits}

        # Compute entropy if requested
        if compute_entropy:
            entropy_mask = attention_mask if attention_mask is not None else None
            entropy = self.chunked_entropy_from_logits_fn(
                logits,
                requires_grad=entropy_requires_grad,
                attention_mask=entropy_mask,
            )
            output["entropy"] = entropy

        # Add router logits for MoE auxiliary loss
        router_logits = self._collect_router_logits()
        if router_logits is not None:
            output["router_logits"] = router_logits

        # Extract action log probs from the end of sequence
        if isinstance(num_actions, list):
            if len(num_actions) == 1:
                num_actions = num_actions[0]
            else:
                # Variable length actions per batch item - need per-sample slicing
                import numpy as np
                num_actions_arr = np.array(num_actions)
                # For variable lengths, we take max and mask invalid positions
                max_actions = int(num_actions_arr.max())
                action_log_probs = log_probs[:, -max_actions - 1: -1]
                if return_output:
                    return action_log_probs, output
                return action_log_probs

        # Action log probs: positions from -num_actions-1 to -1
        # This extracts log probs for predicting the action tokens
        action_log_probs = log_probs[:, -num_actions - 1: -1]

        if return_output:
            return action_log_probs, output
        return action_log_probs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]:
        """Generate tokens autoregressively.

        Args:
            input_ids: Prompt token IDs [batch, prompt_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling (0 = disabled)
            do_sample: Whether to sample (False = greedy)
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID
            attention_mask: Input attention mask

        Returns:
            Tuple of:
                - sequences: Generated sequences [batch, prompt_len + generated_len]
                - attention_mask: Attention mask for full sequences
                - action_mask: Mask indicating valid action positions
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        prompt_len = input_ids.size(1)

        # Initialize sequences with input
        sequences = input_ids.clone()

        # Get EOS token from config if not provided
        if eos_token_id is None:
            eos_token_id = getattr(self.model.config, 'eos_token_id', 199999)
        if pad_token_id is None:
            pad_token_id = eos_token_id

        # Track which sequences are finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            # Get logits for next token
            logits = self.model(sequences)  # [batch, seq_len, vocab]
            next_token_logits = logits[:, -1, :]  # [batch, vocab]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            if do_sample:
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(
                        next_token_logits, top_k
                    )[0][..., -1, None]
                    next_token_logits = next_token_logits.masked_fill(
                        indices_to_remove, float('-inf')
                    )

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = False

                    # Scatter back to original indices
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits = next_token_logits.masked_fill(
                        indices_to_remove, float('-inf')
                    )

                # Sample from the distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                # Greedy decoding
                next_tokens = next_token_logits.argmax(dim=-1)

            # Replace tokens for finished sequences with pad
            next_tokens = torch.where(
                finished, torch.full_like(next_tokens, pad_token_id), next_tokens
            )

            # Update finished status
            finished = finished | (next_tokens == eos_token_id)

            # Append to sequences
            sequences = torch.cat(
                [sequences, next_tokens.unsqueeze(-1)], dim=1
            )

            # Early stopping if all sequences are finished
            if finished.all():
                break

        # Create attention and action masks
        return self.process_sequences(sequences, prompt_len, eos_token_id, pad_token_id)

    def process_sequences(
        self,
        sequences: torch.Tensor,
        input_len: int,
        eos_token_id: int,
        pad_token_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process generated sequences to create attention and action masks.

        Args:
            sequences: Generated sequence tensor [batch, total_len]
            input_len: Length of the input prompt
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID

        Returns:
            Tuple of (sequences, attention_mask, action_mask)
        """
        # Create attention mask: 1 for valid tokens, 0 for pad/after EOS
        attention_mask = (
            sequences.ne(eos_token_id) & sequences.ne(pad_token_id)
        ).to(dtype=torch.long)
        seq_length = attention_mask.size(1)

        # Find position of last valid token
        eos_indices = seq_length - attention_mask.long().fliplr().argmax(
            dim=1, keepdim=True
        ).clamp(min=1)

        # Handle EOS in middle of prompt
        first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
        mask = torch.arange(seq_length, device=sequences.device).unsqueeze(0).expand(
            sequences.size(0), -1
        )
        attention_mask = (
            (mask >= first_token_indices) & (mask <= eos_indices)
        ).to(dtype=torch.long)

        # Create action mask for RL
        state_seq = sequences[:, input_len - 1: -1]
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        action_mask[:, 0] = 1  # First action is always valid

        return sequences, attention_mask, action_mask

    # =========================================================================
    # NMoEModelInterface Implementation
    # =========================================================================

    def forward_with_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        action_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning log probabilities for RL training.

        Args:
            input_ids: Full sequence token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            action_ids: Action token IDs for log prob computation

        Returns:
            Tuple of (log_probs, entropy)
        """
        logits = self.model(input_ids)

        # Compute log probs
        if action_ids is not None:
            log_probs = logprobs_from_logits(logits, action_ids, inplace_backward=True)
        else:
            # Default: use input_ids shifted by 1 as targets
            targets = torch.roll(input_ids, shifts=-1, dims=1)
            log_probs = logprobs_from_logits(logits, targets, inplace_backward=True)

        # Compute entropy
        entropy = self.chunked_entropy_from_logits_fn(
            logits,
            requires_grad=True,
            attention_mask=attention_mask,
        )

        return log_probs, entropy

    def refresh_expert_caches(self) -> None:
        """Refresh quantized weight caches after optimizer step.

        Must be called after each optimizer.step() when using FP8/NVFP4
        quantization to update the cached quantized expert weights.
        """
        for module in self.model.modules():
            if isinstance(module, MoE):
                module.refresh_weight_cache()

    @property
    def uses_quantized_experts(self) -> bool:
        """Whether this model uses quantized expert weights (FP8/NVFP4)."""
        dtype = getattr(self.model.config, 'dtype', 'bf16')
        return dtype in ('fp8', 'nvfp4')

    def get_router_aux_loss(self) -> torch.Tensor:
        """Get auxiliary load balancing loss from routers.

        Returns:
            Scalar tensor with auxiliary loss (0 if no MoE layers)
        """
        aux_loss = torch.tensor(0.0, device=self.device)
        moe_layers = self._get_moe_layers()

        if not moe_layers:
            return aux_loss

        for moe in moe_layers:
            if moe.last_aux_loss is not None:
                aux_loss = aux_loss + moe.last_aux_loss

        return aux_loss / max(len(moe_layers), 1)

    def get_expert_load_stats(self) -> Dict[str, torch.Tensor]:
        """Get expert load statistics for monitoring.

        Returns:
            Dict containing 'loads', 'load_mean', 'load_std'
        """
        moe_layers = self._get_moe_layers()

        if not moe_layers:
            return {
                'loads': torch.tensor([]),
                'load_mean': torch.tensor(0.0),
                'load_std': torch.tensor(0.0),
            }

        # Collect loads from all MoE layers
        loads_list = []
        for moe in moe_layers:
            if moe.last_loads is not None:
                loads_list.append(moe.last_loads)

        if not loads_list:
            return {
                'loads': torch.tensor([]),
                'load_mean': torch.tensor(0.0),
                'load_std': torch.tensor(0.0),
            }

        loads = torch.stack(loads_list, dim=0)  # [n_layers, n_experts]

        return {
            'loads': loads,
            'load_mean': loads.mean(),
            'load_std': loads.std(),
        }

    def update_router_biases(self, gamma: float = 0.001) -> None:
        """Update router biases for aux-free load balancing.

        Args:
            gamma: Update rate for bias adjustment
        """
        moe_layers = self._get_moe_layers()
        for moe in moe_layers:
            if moe.last_loads is not None:
                moe.router.update_bias(moe.last_loads, gamma=gamma)

    def _get_moe_layers(self) -> List["MoE"]:
        """Get all MoE layers from the model."""
        moe_layers = []
        for module in self.model.modules():
            if isinstance(module, MoE):
                moe_layers.append(module)
        return moe_layers

    def _collect_router_logits(self) -> Optional[torch.Tensor]:
        """Collect router logits from MoE layers for auxiliary loss."""
        # Note: nmoe stores loads directly, not router logits
        # Return None as aux loss is computed differently
        return None

    # =========================================================================
    # Gradient Checkpointing
    # =========================================================================

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs: Optional[Dict] = None,
    ) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": False}

        self._gradient_checkpointing = True

        # nmoe uses torch.utils.checkpoint.checkpoint internally in TransformerBlock
        # Enable it by setting a flag that the blocks check
        for block in self.model.blocks:
            if hasattr(block, '_use_gradient_checkpointing'):
                block._use_gradient_checkpointing = True

        logger.info("[NMoE] Gradient checkpointing enabled")

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False

        for block in self.model.blocks:
            if hasattr(block, '_use_gradient_checkpointing'):
                block._use_gradient_checkpointing = False

        logger.info("[NMoE] Gradient checkpointing disabled")

    @property
    def is_gradient_checkpointing(self) -> bool:
        """Whether gradient checkpointing is currently enabled."""
        return self._gradient_checkpointing

    # =========================================================================
    # Model Properties
    # =========================================================================

    @property
    def config(self) -> Any:
        """Get model configuration."""
        return self.model.config

    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Get model dtype."""
        return next(self.model.parameters()).dtype

    def get_input_embeddings(self) -> nn.Module:
        """Get the input embedding layer."""
        return self.model.embedding

    def get_output_embeddings(self) -> nn.Module:
        """Get the output (lm_head) embedding layer."""
        return self.model.lm_head

    # =========================================================================
    # Parameter Access
    # =========================================================================

    def param_sets(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        """Get parameter sets for different learning rates.

        Returns:
            Tuple of (expert_params, dense_params)
        """
        return self.model.param_sets()

    def named_parameters_by_type(
        self,
    ) -> Dict[str, List[Tuple[str, nn.Parameter]]]:
        """Get named parameters organized by type.

        Returns:
            Dict with keys: 'expert', 'router', 'attention', 'dense', 'embedding'
        """
        params_by_type = {
            'expert': [],
            'router': [],
            'attention': [],
            'dense': [],
            'embedding': [],
        }

        for name, param in self.model.named_parameters():
            if 'W1' in name or 'W2' in name or 'W3' in name:
                if '.ffn.' in name and 'router' not in name:
                    params_by_type['expert'].append((name, param))
                else:
                    params_by_type['dense'].append((name, param))
            elif 'router' in name or 'gate' in name:
                params_by_type['router'].append((name, param))
            elif 'attn' in name:
                params_by_type['attention'].append((name, param))
            elif 'embedding' in name or 'embed' in name:
                params_by_type['embedding'].append((name, param))
            else:
                params_by_type['dense'].append((name, param))

        return params_by_type

    # =========================================================================
    # State Dict
    # =========================================================================

    def state_dict_for_save(self) -> Dict[str, torch.Tensor]:
        """Get state dict suitable for checkpoint saving."""
        return self.model.state_dict()

    def load_state_dict_from_checkpoint(
        self,
        state_dict: Dict[str, torch.Tensor],
        strict: bool = True,
    ) -> None:
        """Load state dict from checkpoint."""
        self.model.load_state_dict(state_dict, strict=strict)

        # Refresh expert caches after loading
        if self.uses_quantized_experts:
            self.refresh_expert_caches()

    # =========================================================================
    # Reference Model Support (PPO)
    # =========================================================================

    def freeze_for_reference(self) -> "NMoEModelWrapper":
        """Freeze all parameters for use as reference model in PPO.

        This method freezes all model parameters and sets the model to eval mode.
        Use this when creating a reference model that should not be trained.

        Returns:
            self (for method chaining)

        Example:
            >>> ref_model = NMoEModelWrapper(config)
            >>> ref_model.freeze_for_reference()
            >>> # ref_model is now frozen and in eval mode
        """
        self._frozen_for_reference = True

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Set to eval mode
        self.eval()

        logger.info("[NMoE] Model frozen for reference (all parameters frozen, eval mode)")
        return self

    def unfreeze(self) -> "NMoEModelWrapper":
        """Unfreeze all parameters for training.

        Returns:
            self (for method chaining)
        """
        self._frozen_for_reference = False

        # Unfreeze all parameters
        for param in self.model.parameters():
            param.requires_grad = True

        logger.info("[NMoE] Model unfrozen (all parameters trainable)")
        return self

    def freeze_expert_weights(self) -> "NMoEModelWrapper":
        """Freeze only expert weights, keeping dense layers trainable.

        Useful for training only the dense parts of the model while
        keeping expert weights fixed.

        Returns:
            self (for method chaining)
        """
        expert_params, dense_params = self.param_sets()

        for param in expert_params:
            param.requires_grad = False

        logger.info(
            f"[NMoE] Expert weights frozen ({sum(p.numel() for p in expert_params) / 1e6:.2f}M params)"
        )
        return self

    def freeze_dense_weights(self) -> "NMoEModelWrapper":
        """Freeze only dense weights, keeping expert layers trainable.

        Useful for training only the expert parts of the model (e.g., for
        expert specialization tuning).

        Returns:
            self (for method chaining)
        """
        expert_params, dense_params = self.param_sets()

        for param in dense_params:
            param.requires_grad = False

        logger.info(
            f"[NMoE] Dense weights frozen ({sum(p.numel() for p in dense_params) / 1e6:.2f}M params)"
        )
        return self

    @property
    def is_frozen(self) -> bool:
        """Check if model is frozen for reference."""
        return getattr(self, '_frozen_for_reference', False)

    @property
    def trainable_param_count(self) -> int:
        """Get count of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @property
    def frozen_param_count(self) -> int:
        """Get count of frozen parameters."""
        return sum(p.numel() for p in self.model.parameters() if not p.requires_grad)

    def train(self, mode: bool = True) -> "NMoEModelWrapper":
        """Set training mode.

        Overrides nn.Module.train() to prevent unfreezing reference models.

        Args:
            mode: Whether to set training mode (True) or eval mode (False)

        Returns:
            self
        """
        if mode and self.is_frozen:
            logger.warning(
                "[NMoE] Attempted to set frozen reference model to train mode. "
                "Use unfreeze() first if you want to train this model."
            )
            return self

        super().train(mode)
        return self

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def print_trainable_parameters(self) -> None:
        """Print trainable parameter statistics."""
        expert_params, dense_params = self.param_sets()

        expert_count = sum(p.numel() for p in expert_params)
        dense_count = sum(p.numel() for p in dense_params)
        total_count = expert_count + dense_count

        expert_trainable = sum(p.numel() for p in expert_params if p.requires_grad)
        dense_trainable = sum(p.numel() for p in dense_params if p.requires_grad)

        logger.info(
            f"[NMoE] Parameters: {total_count / 1e6:.2f}M total "
            f"({expert_count / 1e6:.2f}M expert, {dense_count / 1e6:.2f}M dense)"
        )
        logger.info(
            f"[NMoE] Trainable: {(expert_trainable + dense_trainable) / 1e6:.2f}M "
            f"({expert_trainable / 1e6:.2f}M expert, {dense_trainable / 1e6:.2f}M dense)"
        )


def _load_from_checkpoint_path(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> Tuple["Transformer", "NMoEConfig", int, int]:
    """Load nmoe model from checkpoint path.

    Handles both single-file and split-format checkpoints:
    - Single file: Direct torch.load of model state dict
    - Split format: rd.pt (dense) + dp_rank_*.pt (experts)

    Args:
        checkpoint_path: Path to checkpoint directory or file.
        device: Device to load model to. Defaults to CUDA if available.

    Returns:
        Tuple of (model, config, step, tokens_seen)

    Raises:
        FileNotFoundError: If checkpoint path doesn't exist.
        ValueError: If checkpoint format is unrecognized.
    """
    from pathlib import Path
    import os

    path = Path(checkpoint_path)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Determine checkpoint format
    if path.is_dir():
        # Check for split format (iter_XXXXXXX directories)
        rd_path = path / 'rd.pt'
        dp_path = path / 'dp_rank_000.pt'

        # Check if this is an iteration directory
        if rd_path.exists():
            # This is an iteration directory with split format
            it_dir = path
        else:
            # This might be a base checkpoint directory
            # Try to find the latest checkpoint
            try:
                from nmoe.checkpoint import read_tracker, iteration_dir
                step = read_tracker(str(path))
                if step > 0:
                    it_dir = Path(iteration_dir(str(path), step))
                    rd_path = it_dir / 'rd.pt'
                    dp_path = it_dir / 'dp_rank_000.pt'
                else:
                    raise ValueError(
                        f"No valid checkpoint found in {checkpoint_path}. "
                        "Expected iter_XXXXXXX directories or rd.pt file."
                    )
            except ImportError:
                # nmoe.checkpoint not available, try common patterns
                iters = sorted([p for p in path.iterdir()
                               if p.is_dir() and p.name.startswith("iter_")],
                              reverse=True)
                if iters:
                    it_dir = iters[0]
                    rd_path = it_dir / 'rd.pt'
                    dp_path = it_dir / 'dp_rank_000.pt'
                else:
                    raise ValueError(
                        f"No valid checkpoint found in {checkpoint_path}. "
                        "Expected iter_XXXXXXX directories."
                    )

        if not rd_path.exists():
            raise FileNotFoundError(f"rd.pt not found in {it_dir}")

        # Load split format checkpoint
        map_location = str(device) if device.type == 'cpu' else f'cuda:{device.index or 0}'

        # Load rd.pt (dense weights and config)
        rd = torch.load(str(rd_path), map_location=map_location, weights_only=False)
        dense_sd = rd.get('model_dense', rd.get('model', {}))
        step = int(rd.get('step', 0))
        tokens_seen = int(rd.get('tokens', 0))

        # Extract config from run_info or checkpoint
        run_info = rd.get('run_info', {})
        config_dict = rd.get('config', None)

        if config_dict is not None:
            # Config saved directly in checkpoint
            config = NMoEConfig(**config_dict)
        elif run_info:
            # Reconstruct config from run_info (partial info)
            config = NMoEConfig(
                dim=run_info.get('H', 2048),
                n_layers=run_info.get('L', 24),
                n_heads=run_info.get('H', 16),  # Default if not saved
                n_routed_experts=run_info.get('E', 8),
                n_activated_experts=run_info.get('K', 2),
                dtype=run_info.get('dtype', 'bf16'),
            )
            logger.warning(
                "[NMoE] Config reconstructed from run_info (may be incomplete). "
                "Consider saving full config in checkpoint."
            )
        else:
            raise ValueError(
                "Checkpoint does not contain config or run_info. "
                "Cannot determine model architecture."
            )

        # Create model
        model = Transformer(config)
        model.init_weights()

        # Load dense weights
        model.load_state_dict(dense_sd, strict=False)

        # Load expert weights from dp_rank_*.pt if available
        if dp_path.exists():
            dp = torch.load(str(dp_path), map_location=map_location, weights_only=False)
            expert_sd = dp.get('model_expert', {})
            if expert_sd:
                model.load_state_dict(expert_sd, strict=False)
        else:
            logger.warning(f"dp_rank_000.pt not found, loading only dense weights")

        model = model.to(device)
        logger.info(
            f"[NMoE] Loaded checkpoint from {it_dir}: "
            f"step={step}, tokens={tokens_seen:,}"
        )

        return model, config, step, tokens_seen

    elif path.is_file() and path.suffix == '.pt':
        # Single file checkpoint
        map_location = str(device) if device.type == 'cpu' else f'cuda:{device.index or 0}'
        ckpt = torch.load(str(path), map_location=map_location, weights_only=False)

        # Handle different checkpoint formats
        if 'model' in ckpt and 'config' in ckpt:
            # Full checkpoint with config
            config = NMoEConfig(**ckpt['config'])
            model = Transformer(config)
            model.init_weights()
            model.load_state_dict(ckpt['model'])
            step = int(ckpt.get('step', 0))
            tokens_seen = int(ckpt.get('tokens', 0))
        elif 'model_dense' in ckpt:
            # Split format rd.pt file
            raise ValueError(
                f"{path} appears to be an rd.pt file. "
                "Pass the parent directory instead."
            )
        else:
            raise ValueError(
                f"Unrecognized checkpoint format in {path}. "
                "Expected 'model' and 'config' keys or split format."
            )

        model = model.to(device)
        logger.info(f"[NMoE] Loaded checkpoint from {path}: step={step}")

        return model, config, step, tokens_seen

    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


def get_nmoe_model_wrapper(
    model_path_or_config: Union[str, "NMoEConfig", "NMoEModelConfig"],
    temperature: float = 1.0,
    use_torch_compile: bool = False,
    gradient_checkpointing: bool = False,
    device_map: Optional[str] = None,
    device: Optional[torch.device] = None,
    **kwargs,
) -> NMoEModelWrapper:
    """Factory function to create NMoEModelWrapper.

    Args:
        model_path_or_config: Path to model checkpoint, or config object.
            If a string path is provided:
            - Directory: loads from split format (rd.pt + dp_rank_*.pt)
            - File: loads from single checkpoint file
        temperature: Temperature for action selection
        use_torch_compile: Whether to torch.compile entropy calculation
        gradient_checkpointing: Enable gradient checkpointing
        device_map: Device placement (for HF API compatibility, not used)
        device: Specific device to load model to

    Returns:
        NMoEModelWrapper instance

    Example:
        >>> # Load from checkpoint directory
        >>> wrapper = get_nmoe_model_wrapper("/path/to/checkpoint")
        >>> # Load from config
        >>> from nmoe.config import Config
        >>> config = Config(dim=2048, n_layers=24, ...)
        >>> wrapper = get_nmoe_model_wrapper(config)
    """
    if not NMOE_AVAILABLE:
        raise ImportError("nmoe package not available")

    if isinstance(model_path_or_config, str):
        # Load from checkpoint path
        model, config, step, tokens_seen = _load_from_checkpoint_path(
            model_path_or_config,
            device=device,
        )
        logger.info(
            f"[NMoE] Creating wrapper from checkpoint: "
            f"step={step}, tokens={tokens_seen:,}"
        )

        wrapper = NMoEModelWrapper(
            model_or_config=model,
            temperature=temperature,
            use_torch_compile=use_torch_compile,
            gradient_checkpointing=gradient_checkpointing,
            **kwargs,
        )

        # Store checkpoint info for reference
        wrapper._checkpoint_step = step
        wrapper._checkpoint_tokens = tokens_seen

        return wrapper

    return NMoEModelWrapper(
        model_or_config=model_path_or_config,
        temperature=temperature,
        use_torch_compile=use_torch_compile,
        gradient_checkpointing=gradient_checkpointing,
        **kwargs,
    )


class NMoECriticWrapper(nn.Module):
    """Critic wrapper for nmoe models for PPO value estimation.

    This wrapper adds a value head on top of the nmoe Transformer to output
    scalar value estimates for each token position. Used for PPO training
    with a value baseline.

    Args:
        model_or_config: Either a pre-initialized nmoe Transformer model,
            an NMoEConfig, or NMoEModelConfig to construct the model.
        value_head_prefix: Prefix for value head parameters (default "value_head").
        init_value_head: Whether to initialize value head weights (default True).
        gradient_checkpointing: Enable gradient checkpointing at init.

    Example:
        >>> from nmoe.config import Config
        >>> config = Config(dim=2048, n_layers=24, n_heads=16, ...)
        >>> critic = NMoECriticWrapper(config)
        >>> values = critic(sequences, num_actions=32, attention_mask=mask)
    """

    def __init__(
        self,
        model_or_config: Union["Transformer", "NMoEConfig", "NMoEModelConfig", Any],
        value_head_prefix: str = "value_head",
        init_value_head: bool = True,
        gradient_checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()

        if not NMOE_AVAILABLE:
            raise ImportError(
                "nmoe package not available. Please install nmoe to use NMoECriticWrapper."
            )

        self._gradient_checkpointing = False
        self.value_head_prefix = value_head_prefix

        # Handle different input types
        if isinstance(model_or_config, nn.Module):
            # Direct model instance
            self.model = model_or_config
        elif hasattr(model_or_config, 'dim') and hasattr(model_or_config, 'n_layers'):
            # nmoe Config or NMoEModelConfig
            self.model = Transformer(model_or_config)
            self.model.init_weights()
        else:
            raise TypeError(
                f"Expected Transformer, NMoEConfig, or NMoEModelConfig, got {type(model_or_config)}"
            )

        # Get hidden dimension from config
        hidden_size = self.model.config.dim

        # Value head: projects hidden states to scalar values
        self.value_head = nn.Linear(hidden_size, 1, bias=False, dtype=torch.bfloat16)

        if init_value_head:
            # Initialize value head with small weights for stable training
            nn.init.normal_(self.value_head.weight, mean=0.0, std=0.01)

        # Enable gradient checkpointing if requested
        if gradient_checkpointing:
            self.gradient_checkpointing_enable()

        logger.info(f"[NMoE Critic] Initialized with hidden_size={hidden_size}")

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: Union[int, List[int]],
        attention_mask: Optional[torch.Tensor] = None,
        return_output: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Forward pass returning value estimates for PPO.

        Args:
            sequences: Input token IDs [batch, seq_len]
            num_actions: Number of action tokens at the end of sequence
            attention_mask: Attention mask [batch, seq_len] (optional)
            return_output: If True, return (values, output_dict)

        Returns:
            values: Value estimates for action positions [batch, num_actions]
            Or (values, output_dict) if return_output=True
        """
        # Get hidden states from the model (before lm_head)
        # nmoe Transformer returns logits, but we need hidden states
        # Access the last hidden state before lm_head
        hidden_states = self._get_last_hidden_state(sequences)

        # Apply value head to get scalar values
        values = self.value_head(hidden_states).squeeze(-1)  # [batch, seq_len]

        output = {"hidden_states": hidden_states}

        # Extract values for action positions
        if isinstance(num_actions, list):
            if len(num_actions) == 1:
                num_actions = num_actions[0]
            else:
                # Variable length actions - take max and return
                import numpy as np
                max_actions = int(np.array(num_actions).max())
                action_values = values[:, -max_actions - 1: -1]
                if return_output:
                    return action_values, output
                return action_values

        # Extract values for action tokens (same positions as log_probs)
        action_values = values[:, -num_actions - 1: -1]

        if return_output:
            return action_values, output
        return action_values

    def _get_last_hidden_state(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get last hidden state before lm_head.

        nmoe Transformer structure:
        - embedding
        - blocks (transformer layers)
        - norm
        - lm_head

        We want the output after norm, before lm_head.
        """
        # Embedding
        x = self.model.embedding(input_ids)

        # Transformer blocks
        for block in self.model.blocks:
            x = block(x)

        # Final normalization
        x = self.model.norm(x)

        return x

    # =========================================================================
    # Gradient Checkpointing
    # =========================================================================

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs: Optional[Dict] = None,
    ) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": False}

        self._gradient_checkpointing = True

        for block in self.model.blocks:
            if hasattr(block, '_use_gradient_checkpointing'):
                block._use_gradient_checkpointing = True

        logger.info("[NMoE Critic] Gradient checkpointing enabled")

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False

        for block in self.model.blocks:
            if hasattr(block, '_use_gradient_checkpointing'):
                block._use_gradient_checkpointing = False

        logger.info("[NMoE Critic] Gradient checkpointing disabled")

    @property
    def is_gradient_checkpointing(self) -> bool:
        """Whether gradient checkpointing is currently enabled."""
        return self._gradient_checkpointing

    # =========================================================================
    # Model Properties
    # =========================================================================

    @property
    def config(self) -> Any:
        """Get model configuration."""
        return self.model.config

    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Get model dtype."""
        return next(self.model.parameters()).dtype

    def refresh_expert_caches(self) -> None:
        """Refresh quantized weight caches after optimizer step."""
        for module in self.model.modules():
            if isinstance(module, MoE):
                module.refresh_weight_cache()

    @property
    def uses_quantized_experts(self) -> bool:
        """Whether this model uses quantized expert weights (FP8/NVFP4)."""
        dtype = getattr(self.model.config, 'dtype', 'bf16')
        return dtype in ('fp8', 'nvfp4')


def get_nmoe_critic_wrapper(
    model_path_or_config: Union[str, "NMoEConfig", "NMoEModelConfig"],
    value_head_prefix: str = "value_head",
    init_value_head: bool = True,
    gradient_checkpointing: bool = False,
    device: Optional[torch.device] = None,
    **kwargs,
) -> NMoECriticWrapper:
    """Factory function to create NMoECriticWrapper.

    Args:
        model_path_or_config: Path to model checkpoint, or config object.
        value_head_prefix: Prefix for value head parameters.
        init_value_head: Whether to initialize value head weights.
        gradient_checkpointing: Enable gradient checkpointing.
        device: Specific device to load model to.

    Returns:
        NMoECriticWrapper instance

    Example:
        >>> critic = get_nmoe_critic_wrapper("/path/to/checkpoint")
        >>> values = critic(sequences, num_actions=32)
    """
    if not NMOE_AVAILABLE:
        raise ImportError("nmoe package not available")

    if isinstance(model_path_or_config, str):
        # Load from checkpoint path
        model, config, step, tokens_seen = _load_from_checkpoint_path(
            model_path_or_config,
            device=device,
        )
        logger.info(
            f"[NMoE Critic] Creating wrapper from checkpoint: "
            f"step={step}, tokens={tokens_seen:,}"
        )

        wrapper = NMoECriticWrapper(
            model_or_config=model,
            value_head_prefix=value_head_prefix,
            init_value_head=init_value_head,
            gradient_checkpointing=gradient_checkpointing,
            **kwargs,
        )

        return wrapper

    return NMoECriticWrapper(
        model_or_config=model_path_or_config,
        value_head_prefix=value_head_prefix,
        init_value_head=init_value_head,
        gradient_checkpointing=gradient_checkpointing,
        **kwargs,
    )
