#!/bin/bash
set -e

masked_read() {
    local variable_name="$1"
    local prompt="$2"
    local value=""
    local char
    printf '%s' "$prompt"
    while IFS= read -r -s -n1 char; do
        if [ -z "$char" ]; then
            break
        fi
        case "$char" in
            $'\177'|$'\b')
                if [ -n "$value" ]; then
                    value="${value%?}"
                    printf '\b \b'
                fi
                ;;
            *)
                value+="$char"
                printf '●'
                ;;
        esac
    done
    printf '\n'
    printf -v "$variable_name" '%s' "$value"
}

if [ -n "${HF_TOKEN:-}" ]; then
    echo "Found HF_TOKEN in the environment; verifying it with Hugging Face..."
else
    if [ ! -t 0 ]; then
        echo "ERROR: HF_TOKEN is not set and no interactive terminal is available." >&2
        exit 1
    fi
    masked_read HF_TOKEN "HF_TOKEN is not set. Paste your Hugging Face token: "
    export HF_TOKEN
fi

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: no Hugging Face token was provided." >&2
    exit 1
fi

HF_ACCOUNT=$(wget -qO- --timeout=20 --header="Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/whoami-v2) || {
    echo "ERROR: Hugging Face rejected the token or token verification could not connect." >&2
    exit 1
}
HF_NAME=$(printf '%s' "$HF_ACCOUNT" | sed -n 's/.*"name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
if [ -z "$HF_NAME" ]; then
    echo "ERROR: Hugging Face returned an unexpected response while verifying the token." >&2
    exit 1
fi
echo "Hugging Face token verified for $HF_NAME."

read -r -p "Permanently terminate this RunPod after 2 hours? [y/N]: " AUTO_STOP
case "$AUTO_STOP" in
    y|Y|yes|YES|Yes)
        if [ -z "${RUNPOD_POD_ID:-}" ]; then
            echo "ERROR: RUNPOD_POD_ID is not set, so automatic termination cannot be scheduled." >&2
            exit 1
        fi
        RUNPOD_KEY="${RUNPOD_API_KEY:-}"
        if [ -z "$RUNPOD_KEY" ]; then
            masked_read RUNPOD_KEY "Paste your RunPod API key: "
        fi
        if [ -z "$RUNPOD_KEY" ]; then
            echo "ERROR: no RunPod API key was provided." >&2
            exit 1
        fi
        if ! RUNPOD_API_KEY="$RUNPOD_KEY" python3 "$(dirname "$0")/setup.py" --auto-stop-worker-check "$RUNPOD_POD_ID"; then
            exit 1
        fi
        RUNPOD_API_KEY="$RUNPOD_KEY" nohup python3 "$(dirname "$0")/setup.py" --auto-stop-worker "$RUNPOD_POD_ID" \
            >/tmp/comfy-runpod-auto-stop.log 2>&1 &
        unset RUNPOD_KEY
        echo "Permanent termination scheduled for pod $RUNPOD_POD_ID in 2 hours."
        ;;
    *)
        echo "Automatic pod termination not scheduled."
        ;;
esac

cd /workspace/runpod-slim/ComfyUI

echo "===== SNAPSHOT BEFORE ANYTHING RUNS ====="  # catches a stale process before touching anything else
ps aux | grep -E "main.py|python" | grep -v grep || echo "nothing running"
echo "---"
ss -ltnp 2>/dev/null | grep ':8188' || echo "nothing on 8188"
echo "---"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
echo "---"
df -h /workspace 2>/dev/null
echo "---"
ls custom_nodes/ 2>/dev/null
echo "==========================================="

if command -v fuser >/dev/null 2>&1; then  # 0) kill old ComfyUI on port 8188 (fuser if installed, else find PID via ss)
    fuser -k 8188/tcp 2>/dev/null || true
else
    PORT_PID=$(ss -ltnp 2>/dev/null | grep ':8188' | grep -oP 'pid=\K[0-9]+' | head -n1)
    [ -n "$PORT_PID" ] && kill -9 "$PORT_PID" 2>/dev/null || true
fi
pkill -9 -f "main.py --listen" 2>/dev/null || true  # also kill by cmdline match, works even if launched via relative path
sleep 2
if ss -ltn 2>/dev/null | grep -q ':8188 '; then
    echo "WARNING: something is still listening on 8188 - check 'ps aux | grep main.py' and kill it manually"
fi

git fetch origin  # 1) update ComfyUI to latest
git branch --set-upstream-to=origin/master master 2>/dev/null || true
git reset --hard origin/master

.venv-cu128/bin/pip install -r requirements.txt -q  # 2) install deps + manager
.venv-cu128/bin/pip install -U comfy-kitchen -q
.venv-cu128/bin/pip install -U --pre comfyui-manager -q

.venv-cu128/bin/pip install sageattention -q  # 2.5) required for --use-sage-attention at launch - if this fails to build, drop that flag later instead of running without the package

cd custom_nodes  # 3) CUSTOM NODES - one block per node, each with its own idempotent clone + requirements.txt install
[ -d ComfyUI_MiniMax_H3_Extender ] || git clone https://github.com/tritant/ComfyUI_MiniMax_H3_Extender.git  # 3a) MiniMax H3 Extender (tritant) - required by red_signal_h3_extender_part01.json
cd ..
.venv-cu128/bin/pip install -r custom_nodes/ComfyUI_MiniMax_H3_Extender/requirements.txt -q
cd custom_nodes
# [ -d <folder-name> ] || git clone <repo-url>.git  # 3b) TODO: Spectrum - repo URL not confirmed yet
# [ -d <folder-name> ] || git clone <repo-url>.git  # 3c) TODO: Prompt Builder - repo URL not confirmed yet
cd ..  # restart ComfyUI after adding anything here - re-run setup.sh, then relaunch with Step C below

BASE=/workspace/runpod-slim/ComfyUI/models  # 4) create model folders
mkdir -p "$BASE/diffusion_models" "$BASE/text_encoders" "$BASE/vae" "$BASE/loras"

WGET_RETRY="--tries=20 --waitretry=45 --random-wait --retry-on-http-error=429,500,502,503,504"  # --tries alone does NOT retry HTTP 4xx by default

read -p "How many downloads at a time? (HF rate-limits bigger bursts - default 2): " DL_CONCURRENCY
DL_CONCURRENCY="${1:-${DL_CONCURRENCY:-2}}"  # optional 1st arg overrides the prompt, e.g. `bash setup.sh 3`
case "$DL_CONCURRENCY" in ''|*[!0-9]*) DL_CONCURRENCY=2 ;; esac  # empty/non-numeric input falls back to 2

DL_URLS=(
    "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors|$BASE/diffusion_models"
    "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors|$BASE/text_encoders"
    "https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors|$BASE/text_encoders"
    "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors|$BASE/vae"
    "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors|$BASE/vae"
    "https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8/resolve/main/flux-2-klein-base-9b-fp8.safetensors|$BASE/diffusion_models"
    "https://huggingface.co/black-forest-labs/FLUX.2-small-decoder/resolve/main/full_encoder_small_decoder.safetensors|$BASE/vae"
    "https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors|$BASE/loras"
)  # 5/6) model + lora downloads - sliding-window pool keeps exactly $DL_CONCURRENCY running at all times

DL_FAILED=0
DL_TOTAL=${#DL_URLS[@]}
DL_NEXT=0
DL_RUNNING=0

dl_launch_next() {  # sliding-window pool - starts one more job from the queue, if any remain
    if [ "$DL_NEXT" -lt "$DL_TOTAL" ]; then
        entry="${DL_URLS[$DL_NEXT]}"
        url="${entry%%|*}"
        dest="${entry##*|}"
        wget -c $WGET_RETRY --show-progress --header="Authorization: Bearer $HF_TOKEN" -P "$dest" "$url" &
        DL_NEXT=$((DL_NEXT + 1))
        DL_RUNNING=$((DL_RUNNING + 1))
    fi
}

while [ "$DL_RUNNING" -lt "$DL_CONCURRENCY" ] && [ "$DL_NEXT" -lt "$DL_TOTAL" ]; do
    dl_launch_next  # fill the pool up to DL_CONCURRENCY to start
done
while [ "$DL_RUNNING" -gt 0 ]; do
    wait -n || DL_FAILED=1  # blocks until ANY one running job finishes, then immediately backfills
    DL_RUNNING=$((DL_RUNNING - 1))
    dl_launch_next
done
if [ "$DL_FAILED" -eq 1 ]; then
    echo "WARNING: at least one download above failed (often a 403 on the gated FLUX.2 repos - see ONE-TIME license section) - re-run setup.sh after fixing it, wget -c will resume the rest"
    echo "=== SETUP.SH DONE WITH ERRORS - not auto-launching ComfyUI, fix the download above first, then bash setup.sh again ==="
    exit 1
fi

echo "=== SETUP.SH DONE - models + custom nodes + lora ready. Launching ComfyUI now (Step C). ==="
cd /workspace/runpod-slim/ComfyUI
exec .venv-cu128/bin/python main.py --listen 0.0.0.0 --port 8188 --enable-cors-header --enable-manager --use-sage-attention
