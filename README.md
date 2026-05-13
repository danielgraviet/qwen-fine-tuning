# Qwen2-1.5B QLoRA Fine-Tuning

QLoRA fine-tuning pipeline for `Qwen/Qwen2-1.5B-Instruct` using PEFT + TRL. The environment is fully reproducible via `uv sync`.

---

## Environment

- **Python**: 3.11
- **CUDA**: 12.4
- **GPU tested on**: NVIDIA A100 80GB PCIe

### Dependency versions

| Package | Version | Role |
|---|---|---|
| `torch` | 2.6.0+cu124 | Core tensor ops / CUDA runtime |
| `torchaudio` | 2.6.0+cu124 | Torch ecosystem (required by torch) |
| `torchvision` | 0.21.0+cu124 | Torch ecosystem (required by torch) |
| `transformers` | 5.8.0 | Model loading, tokenizer, `BitsAndBytesConfig` |
| `peft` | 0.19.1 | LoRA adapter injection via `LoraConfig` / `get_peft_model` |
| `trl` | 1.4.0 | `SFTTrainer` + `SFTConfig` for instruction fine-tuning |
| `accelerate` | 1.13.0 | Mixed-precision training and device placement |
| `bitsandbytes` | 0.49.2 | 4-bit NF4 quantization backend |
| `datasets` | 4.8.5 | Dataset loading and preprocessing |
| `sentencepiece` | 0.2.1 | Tokenizer backend for Qwen |
| `einops` | 0.8.2 | Tensor reshaping used internally by the model |
| `safetensors` | 0.7.0 | Safe model weight serialization |
| `scipy` | 1.17.1 | Numerical utilities (transitive dep) |
| `numpy` | 2.4.4 | Array ops |

### Important version constraints

- `transformers >= 5.x` requires `torch >= 2.5`. Do not downgrade torch below 2.5 or imports will fail at the `transformers.integrations.moe` module.
- `trl >= 1.0` removed `DataCollatorForCompletionOnlyLM`. Use `completion_only_loss=True` in `SFTConfig` with a `prompt`/`completion` dataset format instead.
- `trl >= 1.0` also removed `assistant_only_loss` support for templates without `{% generation %}` markers (Qwen2's built-in template lacks them). Use `completion_only_loss=True` as above.
- PyTorch CUDA wheels are not on PyPI. They are fetched from `https://download.pytorch.org/whl/cu124` — this is wired into `pyproject.toml` via `[[tool.uv.index]]` so `uv sync` handles it automatically.

---

## Setup

```bash
# 1. Install uv (skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create the virtual environment and install all pinned dependencies
uv sync

# 3. Run training
uv run python3 train.py
```

The lockfile (`uv.lock`) pins all 81 transitive dependencies. `uv sync` will produce a bit-for-bit identical environment on any machine with CUDA 12.4.

---

## Project structure

```
.
├── pyproject.toml   # project metadata, pinned deps, pytorch index config
├── uv.lock          # full transitive lockfile (auto-generated, do not edit)
├── train.py         # training entry point
└── README.md
```

---

## How train.py works

### Quantization (QLoRA)

The base model is loaded in **4-bit NF4** quantization via `BitsAndBytesConfig`. This reduces Qwen2-1.5B's VRAM footprint to ~2GB, leaving the rest free for activations and optimizer state.

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

### LoRA adapters

`LoraConfig` injects small trainable weight matrices into the four attention projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`). Only ~1% of parameters are trained; base weights stay frozen.

```python
LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM")
```

### Dataset

Uses the public `tatsu-lab/alpaca` dataset (first 2000 rows) as a demonstration. Each row is converted to a `messages` list `[{role, content}, ...]` so the Qwen chat template can be applied automatically by `SFTTrainer`.

To use your own data, replace the `load_dataset` call and update `format_messages` to return the same `{"messages": [...]}` structure.

### Loss masking

`completion_only_loss=True` in `SFTConfig` masks the prompt tokens so loss is computed only on the assistant's response. The dataset is formatted with separate `prompt` and `completion` columns — `prompt` is the user turn rendered through the Qwen chat template (with `add_generation_prompt=True`), and `completion` is the assistant response + EOS token. This avoids a TRL 1.x incompatibility with Qwen2's chat template, which lacks the `{% generation %}` markers that `assistant_only_loss` requires.

### Training hyperparameters

| Hyperparameter | Value |
|---|---|
| Epochs | 1 |
| Per-device batch size | 4 |
| Gradient accumulation steps | 4 (effective batch = 16) |
| Learning rate | 2e-4 |
| LR scheduler | cosine |
| Warmup ratio | 0.05 |
| Precision | bf16 |
| Max sequence length | 512 |

### Output

The fine-tuned LoRA adapter is saved to `./qwen-finetuned/`. It contains only the adapter weights (not the full base model), so it is small (~50MB for r=16).

---

## Saved artifacts

The trained adapter and a mid-training checkpoint are backed up to Hugging Face Hub and will survive pod deletion:

- **Model weights**: https://huggingface.co/dtgraviet/quen-fine-tuning

---

## Recovering after a pod restart or deletion

The container filesystem is ephemeral. Code is on GitHub; weights are on Hugging Face. To restore a full working environment from scratch:

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 2. Clone the repo and restore the environment
git clone https://github.com/danielgraviet/quen-fine-tuning
cd quen-fine-tuning
uv sync

# 3. Log in to Hugging Face (requires a token with read access)
#    Get one at https://huggingface.co/settings/tokens
hf auth login

# 4. Pull the trained adapter weights back down
hf download dtgraviet/quen-fine-tuning --local-dir ./qwen-finetuned

# 5. Resume training from the saved checkpoint, or run inference
python3 train.py
```

> **Note**: `uv sync` installs into a `.venv` and needs ~2.5 GB of free disk space for
> the PyTorch CUDA wheels. Make sure your pod has sufficient volume storage before running it.
