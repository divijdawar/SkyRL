"""R3 (Rollout Routing Replay) — Replay MoE expert routing from inference during training.

During RL training of MoE models, the router's expert selections can diverge between
inference (rollout generation) and training (forward/backward pass). This causes
training-inference KL divergence to spike, sometimes leading to training instability.

R3 fixes this by capturing the expert routing decisions (topk_ids) during inference
via SGLang and replaying those exact decisions during the training forward pass.

Reference: https://arxiv.org/abs/2510.11370
"""

from contextlib import contextmanager
from typing import List, Optional, Set, Tuple

import torch
import torch.nn as nn

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


# MoE gate class names by architecture. When a module's class name matches one
# of these, we treat it as an MoE gate whose forward output contains topk_idx.
_KNOWN_GATE_CLASS_NAMES: Set[str] = {
    # DeepSeek V2/V3 (HuggingFace transformers)
    "MoEGate",
    "DeepseekV2MoEGate",
    "DeepseekV3MoEGate",
}


def _find_moe_gates(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Find all MoE gate/router modules in the model by class name.

    Returns:
        List of (module_name, module) tuples in module traversal order.
        The order determines the layer index mapping into routed_experts.
    """
    gates = []
    for name, module in model.named_modules():
        if type(module).__name__ in _KNOWN_GATE_CLASS_NAMES:
            gates.append((name, module))
    return gates


def _find_topk_idx_position(output: tuple) -> Optional[int]:
    """Find the position of topk_idx (integer tensor) in the gate output tuple.

    DeepSeek MoEGate returns (topk_idx, topk_weight, aux_loss) where topk_idx
    is the only 2D integer tensor. We detect it by dtype and dimensionality.
    """
    for i, elem in enumerate(output):
        if isinstance(elem, torch.Tensor) and elem.ndim == 2 and elem.dtype in (torch.int32, torch.int64, torch.long):
            return i
    return None


def _make_r3_hook(routed_experts: torch.Tensor, layer_idx: int):
    """Create a forward hook that replays captured routing for one MoE gate layer.

    The hook replaces the gate's topk_idx output for response token positions
    with the expert selections captured during inference. Prompt token positions
    and positions with -1 sentinel (padding/observation tokens) keep free routing.

    Args:
        routed_experts: [batch, response_len, num_moe_layers, top_k] int32 tensor.
        layer_idx: Index of this gate in the model's MoE layer ordering.
    """
    response_len = routed_experts.shape[1]

    def hook(_module, _input, output):
        if not isinstance(output, tuple):
            return output

        idx_pos = _find_topk_idx_position(output)
        if idx_pos is None:
            return output

        topk_idx = output[idx_pos]  # [total_tokens, top_k]
        batch_size = routed_experts.shape[0]
        total_tokens = topk_idx.shape[0]
        top_k = topk_idx.shape[1]

        if total_tokens % batch_size != 0:
            # Sample packing or unexpected layout — skip replay for safety
            return output

        seq_len = total_tokens // batch_size
        if response_len > seq_len:
            # Response is longer than sequence (shouldn't happen) — skip
            return output

        # Captured routing for this layer: [batch, response_len, top_k]
        captured = routed_experts[:, :, layer_idx, :].to(device=topk_idx.device, dtype=topk_idx.dtype)

        # Reshape to [batch, seq_len, top_k] for position-aware replacement
        topk_idx_3d = topk_idx.view(batch_size, seq_len, top_k).clone()

        # Response tokens occupy the last `response_len` positions (left-padded prompts)
        response_slice = topk_idx_3d[:, -response_len:, :]

        # Only replace where captured routing is valid (not -1 sentinel)
        valid_mask = captured >= 0
        topk_idx_3d[:, -response_len:, :] = torch.where(valid_mask, captured, response_slice)

        # Rebuild output tuple with replaced topk_idx
        output_list = list(output)
        output_list[idx_pos] = topk_idx_3d.view(total_tokens, top_k)
        return tuple(output_list)

    return hook


@contextmanager
def r3_routing_replay(model: nn.Module, routed_experts: Optional[torch.Tensor] = None):
    """Context manager that replays captured MoE expert routing during the forward pass.

    When active, registers forward hooks on all detected MoE gate modules that
    replace the computed topk_idx with values captured during inference rollouts.

    This is a no-op when routed_experts is None (non-MoE models or R3 disabled).

    Args:
        model: The model (or model wrapper) to search for MoE gate modules.
        routed_experts: [batch, response_len, num_moe_layers, top_k] int32 tensor
            of expert indices captured during inference. None to disable replay.

    Raises:
        ValueError: If the number of detected MoE gates doesn't match the
            num_moe_layers dimension of routed_experts.
    """
    if routed_experts is None:
        yield
        return

    gates = _find_moe_gates(model)
    if not gates:
        logger.warning(
            "R3: No MoE gate modules found in model. "
            f"Searched for class names: {_KNOWN_GATE_CLASS_NAMES}. "
            "Routing replay will be a no-op."
        )
        yield
        return

    num_moe_layers = routed_experts.shape[2]
    if len(gates) != num_moe_layers:
        raise ValueError(
            f"R3: Found {len(gates)} MoE gate modules but routed_experts has "
            f"{num_moe_layers} layers. These must match. "
            f"Gate modules found: {[name for name, _ in gates]}"
        )

    batch, resp_len, _, top_k = routed_experts.shape
    logger.debug(
        f"R3: Replaying expert routing for {len(gates)} MoE layers, "
        f"batch={batch}, resp_len={resp_len}, top_k={top_k}"
    )

    # Register hooks — each gate gets a hook for its corresponding layer index
    handles = []
    for layer_idx, (_name, gate) in enumerate(gates):
        handle = gate.register_forward_hook(_make_r3_hook(routed_experts, layer_idx))
        handles.append(handle)

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
