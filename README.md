# comfy-runpod-setup

Cold-start script for a RunPod pod running ComfyUI + MiniMax H3 + FLUX.2 Klein for the Red Signal project. Handles the full sequence from a fresh pod to a running ComfyUI instance: kills any stale process, updates ComfyUI core, installs dependencies, clones the required custom node, downloads all models/LoRA in parallel, then launches ComfyUI.

Two equivalent versions are included — `setup.sh` (bash) and `setup.py` (Python). They do the same thing; use whichever is already available on the pod.

## Requirements

- A RunPod pod with ComfyUI already installed at `/workspace/runpod-slim/ComfyUI`, with a Python venv at `.venv-cu128/`
- A Hugging Face token (see **Gated repos** below). Normally, set `HF_TOKEN` via a RunPod Secret referenced as `{{ RUNPOD_SECRET_yourSecretName }}` in the pod's Environment Variables. If `HF_TOKEN` is absent, the script securely prompts for it in the terminal and displays a `●` for each hidden character entered.
- `wget`, `git`, `pip` available in the venv (standard on the `runpod-slim` image)
- Optional automatic stop requires the pod-provided `RUNPOD_POD_ID` environment variable. The script uses `RUNPOD_API_KEY` or securely prompts for a RunPod API key with `●` masking, then verifies it directly with RunPod's API before scheduling the stop.

## Usage

Clone this repo directly onto the pod and run it:

```bash
git clone <this-repo-url>
cd comfy-runpod---setup
bash setup.sh [concurrency]      # or: python3 setup.py [concurrency]
```

`concurrency` is optional and defaults to **2** — the number of model downloads allowed to run at the same time. Hugging Face rate-limits (HTTP 429) bursts of parallel downloads from RunPod's shared datacenter IPs; 2 has proven reliable in testing. Raise it if you want to try faster, or drop to 1 if you still see 429s.

Before making any changes, the script validates the environment or manually entered token with Hugging Face. An invalid token or a verification/network failure stops setup immediately.

The script also asks whether it should automatically stop the pod after two hours. Answering `yes` verifies that `runpodctl` can access the current pod, then starts a detached timer. This **stops** the pod and releases its GPU while preserving `/workspace`; it does not permanently terminate/delete the pod. The timer begins when you answer the prompt.

On success, the script ends by launching ComfyUI itself (`--listen 0.0.0.0 --port 8188 --enable-cors-header --enable-manager --use-sage-attention`) — no separate launch step needed. If any download fails, it prints a warning and exits without launching, so you never end up running against incomplete models without noticing.

### Relaunching without rerunning setup

If ComfyUI crashes and you just want to restart it (not rerun the whole setup):

```bash
cd /workspace/runpod-slim/ComfyUI
.venv-cu128/bin/python main.py --listen 0.0.0.0 --port 8188 --enable-cors-header --enable-manager --use-sage-attention
```

## What it does, in order

1. Loads `HF_TOKEN` from the environment or securely prompts for it, then verifies it with Hugging Face. Setup stops if verification fails.
2. Offers to schedule an automatic pod stop in two hours.
3. Prints a snapshot (running processes, port 8188 status, GPU, disk, custom node folder) — useful for spotting a stale process before touching anything.
4. Kills anything already listening on port 8188.
5. `git fetch` + `reset --hard origin/master` on the ComfyUI checkout.
6. Installs/upgrades `requirements.txt`, `comfy-kitchen`, `comfyui-manager`, and `sageattention`.
7. Clones [`tritant/ComfyUI_MiniMax_H3_Extender`](https://github.com/tritant/ComfyUI_MiniMax_H3_Extender) into `custom_nodes/` if not already present, and installs its requirements.
8. Downloads all models + the Turbo LoRA in parallel (see **Models downloaded** below), with retry/backoff tuned for Hugging Face's rate limiting (`--retry-on-http-error=429,...`, staggered via a concurrency-limited pool, resumable with `-c` if interrupted).
9. Launches ComfyUI.

## Models downloaded

| File | Destination |
|---|---|
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/` |
| `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | `models/text_encoders/` |
| `qwen_3_8b_fp8mixed.safetensors` | `models/text_encoders/` |
| `minimax_h3_video_vae_fp16.safetensors` | `models/vae/` |
| `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` |
| `flux-2-klein-base-9b-fp8.safetensors` \* | `models/diffusion_models/` |
| `full_encoder_small_decoder.safetensors` \* | `models/vae/` |
| `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | `models/loras/` |

\* Gated repos — see below.

## Gated repos (one-time)

Two files come from Hugging Face repos that require accepting a license before they'll download, even with a valid token. While logged into the HF account `HF_TOKEN` belongs to, visit each and click **Agree and access repository**:

- https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8
- https://huggingface.co/black-forest-labs/FLUX.2-small-decoder

Skipping this makes those two downloads 403 even with a correct token.

## Known issue: CUDA driver / torch version mismatch

`pip install -r requirements.txt` installs whatever torch build satisfies ComfyUI's current version pin from the default PyPI index — this isn't guaranteed to match the CUDA version your specific pod's GPU driver supports, and has caused `RuntimeError: The NVIDIA driver on your system is too old` on more than one pod. If you hit this after running setup:

```bash
.venv-cu128/bin/pip show torch | grep Version
.venv-cu128/bin/python -c "import torch; print(torch.version.cuda)"
nvidia-smi | grep "CUDA Version"
```

If `torch.version.cuda` is higher than what `nvidia-smi` reports the driver supports, reinstall the same torch version pinned to the matching CUDA wheel from `download.pytorch.org/whl/<cuXXX>` instead of the default index. This isn't automated yet — worth fixing in the script if it keeps recurring across pods.

## Turbo LoRA

`minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` (from `lightx2v/Minimax-h3-Turbo`) is downloaded automatically. It's a 4-step model — when using it in a workflow, drop the sampler's `steps` to match (4, not the default 20).
