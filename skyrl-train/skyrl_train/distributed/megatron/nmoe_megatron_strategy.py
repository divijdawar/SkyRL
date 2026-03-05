"""
NMoE-Megatron Integration Strategy.

This module provides a distributed training strategy that combines:
- Megatron's parallel state infrastructure (TP, PP, EP, CP)
- NMoE's RDEP (Routed-then-Dispatched Expert Parallelism) for efficient MoE
- Weight synchronization with SGLang inference engines
"""

from __future__ import annotations

import os
import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import distributed as dist
from torch import optim
from jaxtyping import Float

from skyrl_train.distributed.megatron.megatron_strategy import MegatronStrategy
from skyrl_train.distributed.utils import ModelOrModelOptimPair
from skyrl_train.utils.io import io

try:
    import megatron.core.parallel_state as mpu
    from megatron.core.optimizer import DistributedOptimizer
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
    MEGATRON_AVAILABLE = True
except ImportError:
    MEGATRON_AVAILABLE = False
    mpu = None

try:
    from nmoe.rdep import Rdep
    from nmoe.config import Config as NMoEConfig
    from nmoe.checkpoint import save_checkpoint as nmoe_save_checkpoint
    from nmoe.checkpoint import load_checkpoint as nmoe_load_checkpoint
    NMOE_AVAILABLE = True
except ImportError:
    NMOE_AVAILABLE = False
    Rdep = None


class NMoEMegatronConfig:
    """Configuration for NMoE-Megatron integration."""

    def __init__(
        self,
        # Megatron parallel sizes
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        context_parallel_size: int = 1,
        # NMoE/Expert parallel sizes
        expert_model_parallel_size: int = 1,
        expert_tensor_parallel_size: int = 1,
        # RDEP configuration
        rdep_profile: str = "bf16",  # "bf16", "fp8", "nvfp4"
        rdep_capacity: int = 65536,
        rdep_use_persistent_dispatch: bool = True,
        # Checkpointing
        checkpoint_format: str = "nmoe",  # "nmoe" or "megatron"
        save_expert_weights_separately: bool = True,
        # Weight sync with inference
        enable_inference_sync: bool = True,
        inference_sync_frequency: int = 1,
        # Memory optimization
        offload_experts_to_cpu: bool = False,
        gradient_checkpointing: bool = False,
    ):
        self.tensor_model_parallel_size = tensor_model_parallel_size
        self.pipeline_model_parallel_size = pipeline_model_parallel_size
        self.context_parallel_size = context_parallel_size
        self.expert_model_parallel_size = expert_model_parallel_size
        self.expert_tensor_parallel_size = expert_tensor_parallel_size
        self.rdep_profile = rdep_profile
        self.rdep_capacity = rdep_capacity
        self.rdep_use_persistent_dispatch = rdep_use_persistent_dispatch
        self.checkpoint_format = checkpoint_format
        self.save_expert_weights_separately = save_expert_weights_separately
        self.enable_inference_sync = enable_inference_sync
        self.inference_sync_frequency = inference_sync_frequency
        self.offload_experts_to_cpu = offload_experts_to_cpu
        self.gradient_checkpointing = gradient_checkpointing


class NMoEMegatronStrategy(MegatronStrategy):
    """
    Distributed training strategy combining Megatron parallelism with NMoE's RDEP.

    This strategy provides:
    1. Tensor/Pipeline/Context parallelism from Megatron
    2. Expert parallelism using NMoE's RDEP infrastructure
    3. Efficient MoE computation with IPC/NVSHMEM dispatch
    4. Weight synchronization with SGLang inference engines
    5. Flexible checkpointing (NMoE or Megatron format)

    Usage:
        config = NMoEMegatronConfig(
            tensor_model_parallel_size=2,
            expert_model_parallel_size=8,
            rdep_profile="nvfp4",
        )
        strategy = NMoEMegatronStrategy(config)
        strategy.setup_distributed()
    """

    def __init__(
        self,
        config: NMoEMegatronConfig,
        optimizer_config: Optional[Any] = None,
        seed: int = 42,
        is_lora: bool = False,
    ) -> None:
        if not MEGATRON_AVAILABLE:
            raise ImportError("Megatron-Core is required for NMoEMegatronStrategy")
        if not NMOE_AVAILABLE:
            raise ImportError("nmoe package is required for NMoEMegatronStrategy")

        # Convert NMoEMegatronConfig to Megatron config format
        megatron_config = self._create_megatron_config(config)
        super().__init__(
            megatron_config=megatron_config,
            optimizer_config=optimizer_config,
            seed=seed,
            is_lora=is_lora,
        )

        self.nmoe_config = config
        self._rdep_instances: Dict[str, Rdep] = {}
        self._inference_engine_client = None
        self._weight_version = 0

    def _create_megatron_config(self, config: NMoEMegatronConfig) -> Any:
        """Create a Megatron config object from NMoEMegatronConfig."""
        class MegatronConfig:
            pass

        megatron_config = MegatronConfig()
        megatron_config.tensor_model_parallel_size = config.tensor_model_parallel_size
        megatron_config.pipeline_model_parallel_size = config.pipeline_model_parallel_size
        megatron_config.context_parallel_size = config.context_parallel_size
        megatron_config.expert_model_parallel_size = config.expert_model_parallel_size
        megatron_config.expert_tensor_parallel_size = config.expert_tensor_parallel_size
        return megatron_config

    def setup_distributed(self, timeout: timedelta = timedelta(minutes=30)) -> None:
        """Initialize distributed state with Megatron + NMoE RDEP."""
        # Initialize Megatron parallel state
        super().setup_distributed(timeout=timeout)

        # Initialize NMoE RDEP expert parallel group
        self._setup_nmoe_ep_group()

        self.print(f"NMoE-Megatron distributed setup complete:")
        self.print(f"  TP={self.nmoe_config.tensor_model_parallel_size}")
        self.print(f"  PP={self.nmoe_config.pipeline_model_parallel_size}")
        self.print(f"  CP={self.nmoe_config.context_parallel_size}")
        self.print(f"  EP={self.nmoe_config.expert_model_parallel_size}")
        self.print(f"  ETP={self.nmoe_config.expert_tensor_parallel_size}")
        self.print(f"  RDEP profile={self.nmoe_config.rdep_profile}")

    def _setup_nmoe_ep_group(self) -> None:
        """Set up NMoE's expert parallel process group."""
        # Get the expert parallel group from Megatron
        self._ep_group = mpu.get_expert_model_parallel_group()
        self._ep_rank = mpu.get_expert_model_parallel_rank()
        self._ep_world_size = mpu.get_expert_model_parallel_world_size()

        # Store for RDEP initialization
        self._nmoe_ep_config = {
            "ep_group": self._ep_group,
            "ep_rank": self._ep_rank,
            "ep_world_size": self._ep_world_size,
            "profile": self.nmoe_config.rdep_profile,
            "capacity": self.nmoe_config.rdep_capacity,
        }

    def get_rdep_instance(
        self,
        name: str,
        dim: int,
        n_local_experts: int,
        topk: int,
    ) -> "Rdep":
        """Get or create an RDEP instance for a specific MoE layer.

        Args:
            name: Unique identifier for this MoE layer
            dim: Hidden dimension
            n_local_experts: Number of local experts on this rank
            topk: Number of experts activated per token

        Returns:
            Configured Rdep instance
        """
        if name not in self._rdep_instances:
            self._rdep_instances[name] = Rdep(
                dim=dim,
                n_local=n_local_experts,
                topk=topk,
                profile=self.nmoe_config.rdep_profile,
                capacity=self.nmoe_config.rdep_capacity,
                ep_group=self._ep_group,
            )
        return self._rdep_instances[name]

    def clear_rdep_cache(self) -> None:
        """Clear all cached RDEP instances."""
        for rdep in self._rdep_instances.values():
            if hasattr(rdep, "release"):
                rdep.release()
        self._rdep_instances.clear()
        torch.cuda.empty_cache()

    def backward(
        self,
        loss: torch.Tensor,
        model: nn.Module,
        optimizer: optim.Optimizer,
        **kwargs,
    ) -> None:
        """Perform backward pass with NMoE-aware gradient handling."""
        # Pipeline parallelism backward with Megatron schedule
        if self.nmoe_config.pipeline_model_parallel_size > 1:
            # Use Megatron's pipeline schedule for backward
            # The forward pass already captured activations, backward propagates through stages
            from megatron.core import parallel_state
            from megatron.core.pipeline_parallel import get_forward_backward_func

            # Get the backward function based on virtual pipeline config
            if hasattr(self, '_forward_backward_func') and self._forward_backward_func is not None:
                # Use stored forward_backward_func from forward pass
                # For pure backward, we just call loss.backward() as Megatron handles the schedule
                loss.backward()
            else:
                # Fallback: standard backward (pipeline stages sync via NCCL)
                loss.backward()

            # Sync gradients across pipeline stages for RDEP expert weights
            self._sync_expert_gradients_across_pp(model)
        else:
            # Standard backward with gradient accumulation
            loss.backward()

    def _sync_expert_gradients_across_pp(self, model: nn.Module) -> None:
        """Synchronize expert gradients across pipeline parallel stages.

        In pipeline parallelism, each stage only has a subset of layers.
        Expert weights need gradient synchronization across stages that share experts.
        """
        from megatron.core import parallel_state

        if not parallel_state.is_pipeline_last_stage() and not parallel_state.is_pipeline_first_stage():
            # Middle stages: no special handling needed, grads flow through pipeline
            return

        # For first/last stages with shared experts, sync via coalesced all-reduce
        # to avoid one collective launch per parameter tensor.
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        bucket_cap_mb = max(1, int(os.getenv("SKYRL_PP_EXPERT_GRAD_COALESCE_MB", "64")))
        bucket_cap_bytes = bucket_cap_mb * 1024 * 1024
        by_dtype: Dict[torch.dtype, List[torch.Tensor]] = {}
        for name, param in model.named_parameters():
            if param.grad is None or ('W1' not in name and 'W2' not in name and 'W3' not in name):
                continue
            grad = param.grad
            if grad.is_sparse:
                torch.distributed.all_reduce(
                    grad,
                    op=torch.distributed.ReduceOp.AVG,
                    group=pp_group,
                )
                continue
            by_dtype.setdefault(grad.dtype, []).append(grad)

        for grads in by_dtype.values():
            bucket: List[torch.Tensor] = []
            bucket_bytes = 0
            for grad in grads:
                grad_bytes = int(grad.numel() * grad.element_size())
                if bucket and bucket_bytes + grad_bytes > bucket_cap_bytes:
                    if hasattr(torch.distributed, "all_reduce_coalesced"):
                        torch.distributed.all_reduce_coalesced(
                            bucket,
                            op=torch.distributed.ReduceOp.AVG,
                            group=pp_group,
                        )
                    else:
                        for tensor in bucket:
                            torch.distributed.all_reduce(
                                tensor,
                                op=torch.distributed.ReduceOp.AVG,
                                group=pp_group,
                            )
                    bucket = []
                    bucket_bytes = 0
                bucket.append(grad)
                bucket_bytes += grad_bytes
            if bucket:
                if hasattr(torch.distributed, "all_reduce_coalesced"):
                    torch.distributed.all_reduce_coalesced(
                        bucket,
                        op=torch.distributed.ReduceOp.AVG,
                        group=pp_group,
                    )
                else:
                    for tensor in bucket:
                        torch.distributed.all_reduce(
                            tensor,
                            op=torch.distributed.ReduceOp.AVG,
                            group=pp_group,
                        )

    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model: nn.Module,
        scheduler: Optional[Any],
        name: str = "model",
        **kwargs,
    ) -> Optional[Float[torch.Tensor, "1"]]:
        """Perform optimizer step and optionally sync with inference engines."""
        # Get gradient norm and step
        grad_norm = super().optimizer_step(
            optimizer=optimizer,
            model=model,
            scheduler=scheduler,
            name=name,
            **kwargs,
        )

        # Update weight version
        self._weight_version += 1

        # Sync with inference engines if enabled
        if (
            self.nmoe_config.enable_inference_sync
            and self._weight_version % self.nmoe_config.inference_sync_frequency == 0
        ):
            self._sync_to_inference_engines(model)

        return grad_norm

    def _sync_to_inference_engines(self, model: nn.Module) -> None:
        """Synchronize model weights to SGLang inference engines."""
        if self._inference_engine_client is None:
            return

        try:
            # Collect model parameters
            state_dict = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    state_dict[name] = param.data

            # Send to inference engines
            self._inference_engine_client.update_weights(
                state_dict,
                version=self._weight_version,
            )
            self.print(f"Synced weights to inference engines (version {self._weight_version})")
        except Exception as e:
            self.print(f"Warning: Failed to sync weights to inference engines: {e}")

    def set_inference_engine_client(self, client: Any) -> None:
        """Set the inference engine client for weight synchronization."""
        self._inference_engine_client = client

    def save_checkpoint(
        self,
        model: nn.Module,
        ckpt_dir: str,
        node_local_rank: int,
        optimizer: Optional[DistributedOptimizer] = None,
        scheduler: Optional[OptimizerParamScheduler] = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        """Save checkpoint in NMoE or Megatron format."""
        if self.nmoe_config.checkpoint_format == "nmoe":
            self._save_nmoe_checkpoint(
                model=model,
                ckpt_dir=ckpt_dir,
                node_local_rank=node_local_rank,
                optimizer=optimizer,
                scheduler=scheduler,
                tokenizer=tokenizer,
            )
        else:
            # Use Megatron format
            super().save_checkpoint(
                model=model,
                ckpt_dir=ckpt_dir,
                node_local_rank=node_local_rank,
                optimizer=optimizer,
                scheduler=scheduler,
                tokenizer=tokenizer,
            )

    def _save_nmoe_checkpoint(
        self,
        model: nn.Module,
        ckpt_dir: str,
        node_local_rank: int,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        """Save checkpoint in NMoE format (compatible with SGLang loading)."""
        # Create checkpoint directory
        if node_local_rank == 0:
            io.makedirs(ckpt_dir, exist_ok=True)
        dist.barrier()

        # Extract model weights
        unwrapped_model = model
        while hasattr(unwrapped_model, "module"):
            unwrapped_model = unwrapped_model.module

        # Determine this rank's role
        dp_rank = mpu.get_data_parallel_rank()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        ep_rank = mpu.get_expert_model_parallel_rank()

        with io.local_work_dir(ckpt_dir) as work_dir:
            if self.nmoe_config.save_expert_weights_separately:
                # Save non-expert weights (shared across all ranks)
                if dp_rank == 0 and pp_rank == 0:
                    shared_weights = {}
                    expert_weights = {}

                    for name, param in unwrapped_model.named_parameters():
                        if self._is_expert_param(name):
                            expert_weights[name] = param.data.cpu()
                        else:
                            shared_weights[name] = param.data.cpu()

                    # Save shared weights (only rank 0 in each EP group)
                    if ep_rank == 0 and tp_rank == 0:
                        shared_path = os.path.join(work_dir, "rd.pt")
                        torch.save({
                            "model_state": shared_weights,
                            "config": self._get_nmoe_config_dict(unwrapped_model),
                            "weight_version": self._weight_version,
                        }, shared_path)
                        self.print(f"Saved shared weights to {shared_path}")

                    # Save expert weights (each DP/EP rank saves its own)
                    expert_path = os.path.join(work_dir, f"dp_rank_{dp_rank}_ep_{ep_rank}_tp_{tp_rank}.pt")
                    torch.save({
                        "expert_state": expert_weights,
                        "ep_rank": ep_rank,
                        "tp_rank": tp_rank,
                    }, expert_path)
                    self.print(f"Saved expert weights to {expert_path}")
            else:
                # Save all weights together per rank
                if dp_rank == 0:
                    state_dict = {
                        name: param.data.cpu()
                        for name, param in unwrapped_model.named_parameters()
                    }
                    rank_path = os.path.join(
                        work_dir,
                        f"model_tp{tp_rank}_pp{pp_rank}_ep{ep_rank}.pt"
                    )
                    torch.save({
                        "model_state": state_dict,
                        "config": self._get_nmoe_config_dict(unwrapped_model),
                        "weight_version": self._weight_version,
                    }, rank_path)

            # Save optimizer state
            if optimizer is not None and dp_rank == 0:
                opt_path = os.path.join(work_dir, f"optimizer_dp{dp_rank}.pt")
                torch.save(optimizer.state_dict(), opt_path)

            # Save scheduler state
            if scheduler is not None and self.is_rank_0():
                sched_path = os.path.join(work_dir, "scheduler.pt")
                torch.save(scheduler.state_dict(), sched_path)

            # Save RNG state
            rng_path = os.path.join(work_dir, f"rng_rank_{dist.get_rank()}.pt")
            torch.save(self.get_rng_state(), rng_path)

            # Save tokenizer and config (rank 0 only)
            if self.is_rank_0():
                hf_dir = os.path.join(work_dir, "huggingface")
                if self.hf_config is not None:
                    self.save_hf_configs(self.hf_config, hf_dir, tokenizer)

                # Save NMoE config JSON
                config_path = os.path.join(work_dir, "nmoe_config.json")
                with open(config_path, "w") as f:
                    json.dump(self._get_nmoe_config_dict(unwrapped_model), f, indent=2)

        dist.barrier()
        self.print(f"Checkpoint saved to {ckpt_dir}")

    def _is_expert_param(self, name: str) -> bool:
        """Check if a parameter name belongs to an expert."""
        expert_patterns = [
            ".experts.",
            ".ffn.W1",
            ".ffn.W2",
            ".ffn.W3",
            "moe_layer",
            "expert_",
        ]
        return any(pattern in name for pattern in expert_patterns)

    def _get_nmoe_config_dict(self, model: nn.Module) -> Dict[str, Any]:
        """Extract NMoE configuration from model."""
        config_dict = {
            "tensor_model_parallel_size": self.nmoe_config.tensor_model_parallel_size,
            "pipeline_model_parallel_size": self.nmoe_config.pipeline_model_parallel_size,
            "expert_model_parallel_size": self.nmoe_config.expert_model_parallel_size,
            "expert_tensor_parallel_size": self.nmoe_config.expert_tensor_parallel_size,
            "rdep_profile": self.nmoe_config.rdep_profile,
            "rdep_capacity": self.nmoe_config.rdep_capacity,
        }

        # Try to extract model-specific config
        if hasattr(model, "config"):
            model_config = model.config
            if hasattr(model_config, "to_dict"):
                config_dict["model_config"] = model_config.to_dict()
            elif isinstance(model_config, dict):
                config_dict["model_config"] = model_config

        return config_dict

    def load_checkpoint(
        self,
        model: nn.Module,
        ckpt_dir: str,
        optimizer: Optional[DistributedOptimizer] = None,
        scheduler: Optional[OptimizerParamScheduler] = None,
        load_module_strict: bool = True,
        load_optimizer_states: bool = True,
        load_lr_scheduler_states: bool = True,
    ) -> tuple:
        """Load checkpoint from NMoE or Megatron format."""
        if not ckpt_dir or not io.exists(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

        # Detect checkpoint format
        with io.local_read_dir(ckpt_dir) as read_dir:
            if os.path.exists(os.path.join(read_dir, "rd.pt")):
                return self._load_nmoe_checkpoint(
                    model=model,
                    ckpt_dir=ckpt_dir,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    load_module_strict=load_module_strict,
                    load_optimizer_states=load_optimizer_states,
                    load_lr_scheduler_states=load_lr_scheduler_states,
                )
            else:
                return super().load_checkpoint(
                    model=model,
                    ckpt_dir=ckpt_dir,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    load_module_strict=load_module_strict,
                    load_optimizer_states=load_optimizer_states,
                    load_lr_scheduler_states=load_lr_scheduler_states,
                )

    def _load_nmoe_checkpoint(
        self,
        model: nn.Module,
        ckpt_dir: str,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        load_module_strict: bool = True,
        load_optimizer_states: bool = True,
        load_lr_scheduler_states: bool = True,
    ) -> tuple:
        """Load checkpoint in NMoE format."""
        unwrapped_model = model
        while hasattr(unwrapped_model, "module"):
            unwrapped_model = unwrapped_model.module

        dp_rank = mpu.get_data_parallel_rank()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        ep_rank = mpu.get_expert_model_parallel_rank()

        with io.local_read_dir(ckpt_dir) as read_dir:
            # Load shared weights
            shared_path = os.path.join(read_dir, "rd.pt")
            if os.path.exists(shared_path):
                shared_data = torch.load(shared_path, map_location="cpu")
                shared_state = shared_data.get("model_state", {})
                self._weight_version = shared_data.get("weight_version", 0)

                # Load shared weights into model
                missing, unexpected = unwrapped_model.load_state_dict(
                    shared_state, strict=False
                )
                self.print(f"Loaded shared weights (missing: {len(missing)}, unexpected: {len(unexpected)})")

            # Load expert weights for this rank
            expert_path = os.path.join(read_dir, f"dp_rank_{dp_rank}_ep_{ep_rank}_tp_{tp_rank}.pt")
            if os.path.exists(expert_path):
                expert_data = torch.load(expert_path, map_location="cpu")
                expert_state = expert_data.get("expert_state", {})

                # Load expert weights
                missing, unexpected = unwrapped_model.load_state_dict(
                    expert_state, strict=False
                )
                self.print(f"Loaded expert weights for EP rank {ep_rank}")

            # Load optimizer state
            if optimizer is not None and load_optimizer_states:
                opt_path = os.path.join(read_dir, f"optimizer_dp{dp_rank}.pt")
                if os.path.exists(opt_path):
                    opt_state = torch.load(opt_path, map_location="cpu")
                    optimizer.load_state_dict(opt_state)
                    self.print("Loaded optimizer state")

            # Load scheduler state
            if scheduler is not None and load_lr_scheduler_states and self.is_rank_0():
                sched_path = os.path.join(read_dir, "scheduler.pt")
                if os.path.exists(sched_path):
                    sched_state = torch.load(sched_path, map_location="cpu")
                    scheduler.load_state_dict(sched_state)
                    self.print("Loaded scheduler state")

            # Load RNG state
            rng_path = os.path.join(read_dir, f"rng_rank_{dist.get_rank()}.pt")
            if os.path.exists(rng_path):
                rng_state = torch.load(rng_path, map_location="cpu")
                self.load_rng_state(rng_state)

        dist.barrier()
        return ckpt_dir, {}

    def save_hf_model(
        self,
        bridge: Any,
        model: nn.Module,
        output_dir: str,
        tokenizer: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """Save model in HuggingFace format."""
        # Create output directory
        if self.is_rank_0():
            io.makedirs(output_dir, exist_ok=True)
        dist.barrier()

        with io.local_work_dir(output_dir) as work_dir:
            if hasattr(bridge, "save_hf_weights"):
                bridge.save_hf_weights(model, work_dir)
            else:
                # Fallback: save state dict directly
                unwrapped_model = model
                while hasattr(unwrapped_model, "module"):
                    unwrapped_model = unwrapped_model.module

                if self.is_rank_0():
                    state_dict = {
                        name: param.data.cpu()
                        for name, param in unwrapped_model.named_parameters()
                    }
                    torch.save(state_dict, os.path.join(work_dir, "pytorch_model.bin"))

            # Save configs
            if self.is_rank_0():
                self.save_hf_configs(self.hf_config, work_dir, tokenizer)

        dist.barrier()
        self.print(f"Saved HF model to {output_dir}")

    def offload_to_cpu(
        self,
        model: nn.Module,
        optimizer: Optional[Any],
        pin_memory: bool = True,
        non_blocking: bool = True,
        offload_optimizer: bool = True,
        offload_model: bool = True,
    ) -> None:
        """Offload model and optimizer to CPU memory."""
        super().offload_to_cpu(
            model=model,
            optimizer=optimizer,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
            offload_optimizer=offload_optimizer,
            offload_model=offload_model,
        )

        # Also clear RDEP instances to free GPU memory
        if self.nmoe_config.offload_experts_to_cpu:
            self.clear_rdep_cache()

    def backload_to_gpu(
        self,
        model: nn.Module,
        optimizer: Optional[Any],
        non_blocking: bool = True,
        backload_optimizer: bool = True,
        backload_model: bool = True,
    ) -> None:
        """Reload model and optimizer back to GPU."""
        super().backload_to_gpu(
            model=model,
            optimizer=optimizer,
            non_blocking=non_blocking,
            backload_optimizer=backload_optimizer,
            backload_model=backload_model,
        )

    def get_expert_parallel_info(self) -> Dict[str, Any]:
        """Get information about expert parallel configuration."""
        return {
            "ep_rank": self._ep_rank,
            "ep_world_size": self._ep_world_size,
            "rdep_profile": self.nmoe_config.rdep_profile,
            "rdep_capacity": self.nmoe_config.rdep_capacity,
            "num_rdep_instances": len(self._rdep_instances),
        }

    def prepare(
        self, *models_or_model_optim_pairs: ModelOrModelOptimPair
    ) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        """Prepare models for distributed training."""
        # For NMoE, we rely on the model already being set up with RDEP
        # This method is primarily for compatibility with the base interface
        return list(models_or_model_optim_pairs)


def create_nmoe_megatron_strategy(
    tp_size: int = 1,
    pp_size: int = 1,
    cp_size: int = 1,
    ep_size: int = 1,
    etp_size: int = 1,
    rdep_profile: str = "bf16",
    rdep_capacity: int = 65536,
    checkpoint_format: str = "nmoe",
    seed: int = 42,
    is_lora: bool = False,
) -> NMoEMegatronStrategy:
    """Factory function to create NMoEMegatronStrategy with common defaults.

    Args:
        tp_size: Tensor parallel size
        pp_size: Pipeline parallel size
        cp_size: Context parallel size
        ep_size: Expert parallel size
        etp_size: Expert tensor parallel size
        rdep_profile: RDEP quantization profile ("bf16", "fp8", "nvfp4")
        rdep_capacity: RDEP buffer capacity
        checkpoint_format: Checkpoint format ("nmoe" or "megatron")
        seed: Random seed
        is_lora: Whether using LoRA adapters

    Returns:
        Configured NMoEMegatronStrategy instance
    """
    config = NMoEMegatronConfig(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        context_parallel_size=cp_size,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=etp_size,
        rdep_profile=rdep_profile,
        rdep_capacity=rdep_capacity,
        checkpoint_format=checkpoint_format,
    )
    return NMoEMegatronStrategy(
        config=config,
        seed=seed,
        is_lora=is_lora,
    )
