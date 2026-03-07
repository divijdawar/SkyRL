"""Tests for R3 (Rollout Routing Replay) — MoE expert routing replay during training."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from skyrl_train.utils.r3_routing_replay import (
    _find_moe_gates,
    _find_topk_idx_position,
    _make_r3_hook,
    r3_routing_replay,
)
from skyrl_train.dataset.preprocess import pad_and_stack_routed_experts


# ---------------------------------------------------------------------------
# Helpers: mock MoE gate module
# ---------------------------------------------------------------------------


class MoEGate(nn.Module):
    """Mock MoE gate that returns (topk_idx, topk_weight, aux_loss)."""

    def __init__(self, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, hidden_states: torch.Tensor):
        # hidden_states: [total_tokens, hidden_dim]
        total_tokens = hidden_states.shape[0]
        topk_idx = torch.randint(0, self.num_experts, (total_tokens, self.top_k), dtype=torch.int64)
        topk_weight = torch.randn(total_tokens, self.top_k)
        aux_loss = torch.tensor(0.0)
        return topk_idx, topk_weight, aux_loss


class MockMoEModel(nn.Module):
    """Model with MoE gate layers for testing."""

    def __init__(self, num_layers: int = 2, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.Module()
            layer.gate = MoEGate(num_experts=num_experts, top_k=top_k)
            self.layers.append(layer)

    def forward(self, x: torch.Tensor):
        results = []
        for layer in self.layers:
            results.append(layer.gate(x))
        return results


# ---------------------------------------------------------------------------
# Test: context manager no-op when routed_experts is None
# ---------------------------------------------------------------------------


class TestR3ContextManagerNoop:
    def test_noop_when_none(self):
        model = nn.Linear(10, 10)
        x = torch.randn(4, 10)
        expected = model(x)

        with r3_routing_replay(model, None):
            actual = model(x)

        assert torch.equal(expected, actual)

    def test_noop_when_no_gates(self, caplog):
        model = nn.Linear(10, 10)
        routed_experts = torch.zeros(1, 5, 2, 2, dtype=torch.int32)
        x = torch.randn(4, 10)
        expected = model(x)

        with r3_routing_replay(model, routed_experts):
            actual = model(x)

        assert torch.equal(expected, actual)
        assert any("No MoE gate modules found" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Test: hook replays routing decisions
# ---------------------------------------------------------------------------


class TestR3HookReplaysRouting:
    def test_replays_captured_routing(self):
        """Verify that response token positions get the captured routing."""
        batch_size = 2
        num_layers = 2
        top_k = 2
        prompt_len = 3
        response_len = 4
        seq_len = prompt_len + response_len

        model = MockMoEModel(num_layers=num_layers, top_k=top_k)

        # Create captured routing: [batch, response_len, num_layers, top_k]
        captured = torch.randint(0, 8, (batch_size, response_len, num_layers, top_k), dtype=torch.int32)

        # Run forward with R3 replay
        x = torch.randn(batch_size * seq_len, 16)  # [total_tokens, hidden_dim]
        with r3_routing_replay(model, captured):
            results = model(x)

        # Check each layer's gate output
        for layer_idx, (topk_idx, _, _) in enumerate(results):
            topk_3d = topk_idx.view(batch_size, seq_len, top_k)
            response_routing = topk_3d[:, -response_len:, :]
            expected = captured[:, :, layer_idx, :].to(dtype=topk_idx.dtype)
            assert torch.equal(response_routing, expected), (
                f"Layer {layer_idx}: response routing doesn't match captured routing"
            )

    def test_preserves_prompt_routing(self):
        """Verify that prompt token positions keep free (original) routing."""
        batch_size = 1
        num_layers = 1
        top_k = 2
        prompt_len = 3
        response_len = 2
        seq_len = prompt_len + response_len

        model = MockMoEModel(num_layers=num_layers, top_k=top_k)

        # Seed for reproducibility — get the original prompt routing
        torch.manual_seed(42)
        x = torch.randn(batch_size * seq_len, 16)
        torch.manual_seed(42)
        original_results = model(x)
        original_prompt_routing = original_results[0][0].view(batch_size, seq_len, top_k)[:, :prompt_len, :]

        # Run with R3 — prompt positions should stay the same
        captured = torch.randint(0, 8, (batch_size, response_len, num_layers, top_k), dtype=torch.int32)
        torch.manual_seed(42)
        with r3_routing_replay(model, captured):
            results = model(x)

        actual_prompt_routing = results[0][0].view(batch_size, seq_len, top_k)[:, :prompt_len, :]
        assert torch.equal(original_prompt_routing, actual_prompt_routing)

    def test_preserves_padding_sentinel(self):
        """Positions with -1 sentinel in captured routing should keep original routing."""
        batch_size = 1
        num_layers = 1
        top_k = 2
        seq_len = 5
        response_len = 3

        model = MockMoEModel(num_layers=num_layers, top_k=top_k)

        # Captured routing with -1 sentinel at position 1
        captured = torch.tensor([[[[3, 5]], [[-1, -1]], [[7, 1]]]], dtype=torch.int32)
        # Shape: [1, 3, 1, 2]

        torch.manual_seed(0)
        x = torch.randn(batch_size * seq_len, 16)

        # Get original routing at the sentinel position
        torch.manual_seed(0)
        original_results = model(x)
        original_topk = original_results[0][0].view(batch_size, seq_len, top_k)
        original_at_sentinel = original_topk[:, -response_len + 1, :].clone()  # position 1 of response

        # Run with R3
        torch.manual_seed(0)
        with r3_routing_replay(model, captured):
            results = model(x)

        actual_topk = results[0][0].view(batch_size, seq_len, top_k)

        # Position 0 of response should be [3, 5]
        assert torch.equal(actual_topk[:, -response_len, :], torch.tensor([[3, 5]], dtype=torch.int64))

        # Position 1 of response (sentinel) should keep original routing
        assert torch.equal(actual_topk[:, -response_len + 1, :], original_at_sentinel)

        # Position 2 of response should be [7, 1]
        assert torch.equal(actual_topk[:, -response_len + 2, :], torch.tensor([[7, 1]], dtype=torch.int64))


# ---------------------------------------------------------------------------
# Test: gate count mismatch raises ValueError
# ---------------------------------------------------------------------------


class TestR3GateCountMismatch:
    def test_raises_on_mismatch(self):
        model = MockMoEModel(num_layers=2, top_k=2)
        # routed_experts has 3 layers but model has 2 gates
        routed_experts = torch.zeros(1, 5, 3, 2, dtype=torch.int32)

        with pytest.raises(ValueError, match="Found 2 MoE gate modules but routed_experts has 3 layers"):
            with r3_routing_replay(model, routed_experts):
                pass


# ---------------------------------------------------------------------------
# Test: pad_and_stack_routed_experts
# ---------------------------------------------------------------------------


class TestPadAndStackRoutedExperts:
    def test_basic_padding(self):
        num_layers = 2
        top_k = 2
        arr1 = np.ones((3, num_layers, top_k), dtype=np.int32)
        arr2 = np.ones((5, num_layers, top_k), dtype=np.int32) * 2

        result = pad_and_stack_routed_experts([arr1, arr2], max_response_len=5)

        assert result.shape == (2, 5, num_layers, top_k)
        assert result.dtype == torch.int32
        # arr1 is padded with -1 for positions 3,4
        assert (result[0, :3] == 1).all()
        assert (result[0, 3:] == -1).all()
        # arr2 has no padding
        assert (result[1] == 2).all()

    def test_truncation(self):
        arr = np.ones((10, 2, 2), dtype=np.int32)
        result = pad_and_stack_routed_experts([arr], max_response_len=5)
        assert result.shape == (1, 5, 2, 2)
        assert (result == 1).all()

    def test_exact_length(self):
        arr = np.ones((5, 2, 2), dtype=np.int32) * 3
        result = pad_and_stack_routed_experts([arr], max_response_len=5)
        assert result.shape == (1, 5, 2, 2)
        assert (result == 3).all()


# ---------------------------------------------------------------------------
# Test: _find_moe_gates discovers known classes
# ---------------------------------------------------------------------------


class TestFindMoeGates:
    def test_discovers_known_classes(self):
        model = MockMoEModel(num_layers=3)
        gates = _find_moe_gates(model)
        assert len(gates) == 3
        for name, module in gates:
            assert type(module).__name__ == "MoEGate"

    def test_empty_for_non_moe_model(self):
        model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 5))
        gates = _find_moe_gates(model)
        assert len(gates) == 0

    def test_preserves_order(self):
        model = MockMoEModel(num_layers=3)
        gates = _find_moe_gates(model)
        names = [name for name, _ in gates]
        # Should be in module traversal order
        assert names == ["layers.0.gate", "layers.1.gate", "layers.2.gate"]


# ---------------------------------------------------------------------------
# Test: _find_topk_idx_position
# ---------------------------------------------------------------------------


class TestFindTopkIdxPosition:
    def test_finds_integer_2d_tensor(self):
        topk_idx = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
        topk_weight = torch.randn(2, 2)
        aux_loss = torch.tensor(0.0)
        assert _find_topk_idx_position((topk_idx, topk_weight, aux_loss)) == 0

    def test_finds_at_different_position(self):
        topk_weight = torch.randn(2, 2)
        topk_idx = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        aux_loss = torch.tensor(0.0)
        assert _find_topk_idx_position((topk_weight, topk_idx, aux_loss)) == 1

    def test_returns_none_for_no_integer_tensor(self):
        a = torch.randn(2, 2)
        b = torch.randn(2, 2)
        assert _find_topk_idx_position((a, b)) is None

    def test_ignores_1d_integer_tensor(self):
        idx_1d = torch.tensor([0, 1, 2], dtype=torch.int64)
        weight = torch.randn(3, 2)
        assert _find_topk_idx_position((idx_1d, weight)) is None


# ---------------------------------------------------------------------------
# Test: hooks are removed after context exit
# ---------------------------------------------------------------------------


class TestR3HooksCleanup:
    def test_hooks_removed_after_exit(self):
        model = MockMoEModel(num_layers=2, top_k=2)
        routed_experts = torch.zeros(1, 5, 2, 2, dtype=torch.int32)

        # Count hooks before
        gates = _find_moe_gates(model)
        hooks_before = sum(len(g._forward_hooks) for _, g in gates)
        assert hooks_before == 0

        with r3_routing_replay(model, routed_experts):
            hooks_during = sum(len(g._forward_hooks) for _, g in gates)
            assert hooks_during == 2

        hooks_after = sum(len(g._forward_hooks) for _, g in gates)
        assert hooks_after == 0

    def test_hooks_removed_on_exception(self):
        model = MockMoEModel(num_layers=2, top_k=2)
        routed_experts = torch.zeros(1, 5, 2, 2, dtype=torch.int32)
        gates = _find_moe_gates(model)

        with pytest.raises(RuntimeError):
            with r3_routing_replay(model, routed_experts):
                raise RuntimeError("simulated error")

        hooks_after = sum(len(g._forward_hooks) for _, g in gates)
        assert hooks_after == 0
