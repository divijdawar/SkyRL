# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

SkyRL is a monorepo containing a full-stack Reinforcement Learning framework for LLMs, split into four Python packages:

- **`skyrl-train/`** — Core RL training framework (PPO, GRPO, RLOO, REINFORCE++, DAPO, etc.)
- **`skyrl-gym/`** — RL environments (GSM8K, AIME, LiveCodeBench, SQL, search tasks)
- **`skyrl-agent/`** — Agentic layer for long-horizon tasks (SWE, Web Research, MemAgent)
- **`skyrl-tx/`** — Unified training + inference server (JAX-based)

Each package has its own `pyproject.toml` and `uv.lock`. The project uses **`uv`** as the package manager throughout.

## Behaviour
Every implementation must be lean and minimal. You must optimize for simplicity and correctness over speed and features.
When in doubt, you must pause and ask the programmer, never make any assumptions.
You must explain any change you make and ensure it is properly integrated into the repo. Each change should be accompanied by proper test cases.

## Commands

### Linting & Formatting

```bash
# Lint/format all files (from repo root)
bash format.sh
# or equivalently:
pre-commit run --all-files --config .pre-commit-config.yaml
```

Hooks run: `ruff` (with `--fix`), `black` (line-length 120), `gitleaks`. `ruff` excludes `skyrl-agent/`.

### Testing

```bash
# skyrl-train CPU tests
cd skyrl-train && uv run --frozen pytest tests/cpu/

# Run a single test file
cd skyrl-train && uv run --frozen pytest tests/cpu/test_foo.py

# skyrl-gym tests
cd skyrl-gym && uv run --frozen pytest tests/

# skyrl-tx tests (requires --forked)
cd skyrl-tx && uv run --extra tinker --extra dev pytest --forked -s tests --ignore=tests/gpu
```

GPU tests run via Anyscale CI (see `skyrl-train/ci/`). Test markers: `gpu`, `multi_gpu`, `integration`, `slow`, `sglang`, `megatron`.

### Installation

```bash
# skyrl-train (SGLang backend, H100/A100)
cd skyrl-train && uv sync --extra sglang

# skyrl-gym
cd skyrl-gym && pip install -e .

# skyrl-tx
cd skyrl-tx && uv run --extra gpu --extra tinker ...
```

B200/SM100 uses a separate `[b200]` extra with PyTorch nightly.

### Docs Site

```bash
cd docs && npm install && npm run dev
```

## Architecture

### Training Loop (`skyrl-train`)

The main entrypoint is `skyrl_train/entrypoints/main_base.py`, configured via Hydra (`config/ppo_base_config.yaml`).

**Synchronous trainer**: `trainer.py::RayPPOTrainer`
**Fully async trainer**: `fully_async_trainer.py` (separates generation from training)

Data flow:
1. `PromptDataset` → `GeneratorInterface.generate()` (multi-turn rollouts)
2. Generator calls `InferenceEngineClient` (SGLang) for LLM outputs, then steps through `skyrl-gym` environments for rewards
3. Advantages computed, policy updated via `PPORayActorGroup` (FSDP or Megatron workers)
4. New weights synced back to inference engines via `weight_sync/`

### Key Subsystems

**Workers** (`workers/`): Ray-distributed training workers. `fsdp/fsdp_worker.py` for FSDP/FSDP2; `megatron/` for Megatron-LM.

**Inference Engines** (`inference_engines/`): SGLang backend behind a common interface (`InferenceEngineInput`/`InferenceEngineOutput` TypedDicts). Client wrapper at `inference_engine_client.py`.

**Weight Sync** (`weight_sync/`): Three strategies — NCCL broadcast, CUDA IPC (fastest, same-node only), checkpoint-on-disk fallback.

**Algorithms** (`utils/ppo_utils.py`): `PolicyLossRegistry` and `AdvantageEstimatorRegistry` for pluggable algorithms.

**Environments** (`skyrl-gym/envs/`): Registered via `registration.py` (`register()` / `make()`). Base class `BaseTextEnv` for LLM text tasks. Implement `step()`, `reset()`, `close()`.

### Configuration

All training config lives in `skyrl-train/skyrl_train/config/ppo_base_config.yaml` (Hydra/OmegaConf). Key top-level groups:
- `data.*` — dataset paths
- `trainer.*` — model paths, FSDP config, optimizer, algorithm params
- `generator.*` — inference backend, weight sync, speculative decoding, multi-node
- `environment.env_class` — which gym environment to use
- `curriculum.*` — curriculum sampling

Per-experiment overrides in `config/experiment/`.

### External Dependencies

- `verl` pinned from `https://github.com/limenlp/verl.git` (main branch)
- `sglang` expected at `../../sglang/python` (sibling directory, not in repo)
- `nmoe` expected at `../../nmoe` (sibling directory, not in repo)
- Requires **Python 3.12** exactly for `skyrl-train`; 3.10+ for `skyrl-gym`; 3.11+ for `skyrl-tx`

## AI Agent Session Protocol (`skyrl-train/AGENTS.md`)

When working in `skyrl-train/`, follow the mandatory session-end workflow:
1. File issues using the `bd` (beads) tool
2. Run quality gates (lint, tests)
3. Push changes to remote
