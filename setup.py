#!/usr/bin/env python3
"""Python port of setup.sh - cold-start ComfyUI on a fresh RunPod pod.
Usage: python3 setup.py [concurrency]   (concurrency defaults to 2)
"""
import os
import re
import subprocess
import sys
import getpass
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

COMFYUI_DIR = Path("/workspace/runpod-slim/ComfyUI")
VENV_PY = COMFYUI_DIR / ".venv-cu128" / "bin" / "python"
VENV_PIP = COMFYUI_DIR / ".venv-cu128" / "bin" / "pip"
BASE = COMFYUI_DIR / "models"
HF_TOKEN = ""

WGET_RETRY = [
    "--tries=20", "--waitretry=45", "--random-wait",
    "--retry-on-http-error=429,500,502,503,504",  # --tries alone does NOT retry HTTP 4xx by default
]

DOWNLOADS = [  # (url, destination subfolder)
    ("https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors", "diffusion_models"),
    ("https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "text_encoders"),
    ("https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors", "text_encoders"),
    ("https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors", "vae"),
    ("https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors", "vae"),
    ("https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8/resolve/main/flux-2-klein-base-9b-fp8.safetensors", "diffusion_models"),  # gated - accept access once, see note below
    ("https://huggingface.co/black-forest-labs/FLUX.2-small-decoder/resolve/main/full_encoder_small_decoder.safetensors", "vae"),  # gated - same as above
    ("https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", "loras"),
]

GATED_REPOS_NOTE = (
    "ONE-TIME: accept gated repo access (logged into the HF account HF_TOKEN belongs to), or downloads 403:\n"
    "  https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8\n"
    "  https://huggingface.co/black-forest-labs/FLUX.2-small-decoder"
)


def run(cmd, **kwargs):
    display_cmd = [
        "Authorization: Bearer ***" if str(c).startswith("Authorization: Bearer ") else str(c)
        for c in cmd
    ]
    print(f"+ {' '.join(display_cmd)}")
    return subprocess.run(cmd, **kwargs)


def masked_input(prompt: str) -> str:
    """Read a secret from a Unix terminal while showing one dot per character."""
    if not sys.stdin.isatty():
        return getpass.getpass(prompt)

    import termios
    import tty

    secret = []
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    sys.stdout.write(prompt)
    sys.stdout.flush()
    try:
        tty.setraw(fd)
        while True:
            char = sys.stdin.read(1)
            if char in ("\r", "\n"):
                break
            if char == "\x03":
                raise KeyboardInterrupt
            if char in ("\x08", "\x7f"):
                if secret:
                    secret.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue
            if char.isprintable():
                secret.append(char)
                sys.stdout.write("●")
                sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)
        sys.stdout.write("\n")
        sys.stdout.flush()
    return "".join(secret)


def get_and_validate_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        print("Found HF_TOKEN in the environment; verifying it with Hugging Face...")
    else:
        if not sys.stdin.isatty():
            print("ERROR: HF_TOKEN is not set and no interactive terminal is available.", file=sys.stderr)
            sys.exit(1)
        token = masked_input("HF_TOKEN is not set. Paste your Hugging Face token: ").strip()

    if not token:
        print("ERROR: no Hugging Face token was provided.", file=sys.stderr)
        sys.exit(1)

    request = urllib.request.Request(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            account = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            print("ERROR: Hugging Face rejected the token. Check that it is valid and try again.", file=sys.stderr)
        else:
            print(f"ERROR: Hugging Face token verification failed (HTTP {exc.code}).", file=sys.stderr)
        sys.exit(1)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"ERROR: could not verify the Hugging Face token: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Hugging Face token verified for {account.get('name', 'the authenticated account')}.")
    return token


def offer_pod_auto_stop():
    answer = input("Automatically stop this RunPod after 2 hours? [y/N]: ").strip().lower()
    if answer not in ("y", "yes"):
        print("Automatic pod stop not scheduled.")
        return

    pod_id = os.environ.get("RUNPOD_POD_ID", "").strip()
    if not pod_id:
        print("ERROR: RUNPOD_POD_ID is not set, so the automatic stop cannot be scheduled.", file=sys.stderr)
        sys.exit(1)
    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key:
        if not sys.stdin.isatty():
            print("ERROR: RUNPOD_API_KEY is not set and no interactive terminal is available.", file=sys.stderr)
            sys.exit(1)
        api_key = masked_input("Paste your RunPod API key: ").strip()
    if not api_key:
        print("ERROR: no RunPod API key was provided.", file=sys.stderr)
        sys.exit(1)

    try:
        runpod_graphql_request(pod_id, api_key, action="verify")
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    print("RunPod API key and pod access verified.")

    worker_env = os.environ.copy()
    worker_env["RUNPOD_API_KEY"] = api_key
    log = open("/tmp/comfy-runpod-auto-stop.log", "a", encoding="utf-8")
    try:
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--auto-stop-worker", pod_id],
            start_new_session=True,
            stdout=log,
            stderr=log,
            env=worker_env,
        )
    finally:
        log.close()
    print(f"Automatic stop scheduled for pod {pod_id} in 2 hours.")


def runpod_graphql_request(pod_id: str, api_key: str, action: str):
    if action == "verify":
        query = "query AutoStopPods { myself { pods { id } } }"
    elif action == "stop":
        pod_literal = json.dumps(pod_id)
        query = (
            "mutation AutoStopPod { "
            f"podStop(input: {{podId: {pod_literal}}}) {{ id desiredStatus }} "
            "}"
        )
    else:
        raise ValueError(f"unknown RunPod GraphQL action: {action}")

    url = "https://api.runpod.io/graphql?api_key=" + urllib.parse.quote(api_key, safe="")
    request = urllib.request.Request(
        url,
        data=json.dumps({"query": query}).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise RuntimeError("RunPod rejected the API key or it lacks pod permissions.") from exc
        raise RuntimeError(f"RunPod GraphQL request failed (HTTP {exc.code}).") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not connect to the RunPod API: {exc}") from exc

    if payload.get("errors"):
        message = payload["errors"][0].get("message", "unknown GraphQL error")
        raise RuntimeError(f"RunPod GraphQL error: {message}")

    if action == "verify":
        data = payload.get("data") or {}
        myself = data.get("myself") or {}
        pods = myself.get("pods") or []
        if not any(pod.get("id") == pod_id for pod in pods):
            raise RuntimeError(f"RunPod could not find pod {pod_id} under this API-key account.")
    elif not ((payload.get("data") or {}).get("podStop") or {}).get("id"):
        raise RuntimeError("RunPod did not confirm the pod stop request.")

    return payload


def auto_stop_worker(pod_id: str):
    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key:
        print("Automatic stop failed: RUNPOD_API_KEY is missing.", file=sys.stderr)
        return 1
    time.sleep(2 * 60 * 60)
    try:
        runpod_graphql_request(pod_id, api_key, action="stop")
    except RuntimeError as exc:
        print(f"Automatic stop failed: {exc}", file=sys.stderr)
        return 1
    print(f"Automatic stop requested successfully for pod {pod_id}.")
    return 0


def auto_stop_worker_check(pod_id: str):
    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key:
        print("ERROR: RUNPOD_API_KEY is missing.", file=sys.stderr)
        return 1
    try:
        runpod_graphql_request(pod_id, api_key, action="verify")
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("RunPod API key and pod access verified.")
    return 0


def snapshot():
    print("===== SNAPSHOT BEFORE ANYTHING RUNS =====")
    try:
        ps = subprocess.run(["ps", "aux"], capture_output=True, text=True).stdout
        matches = [l for l in ps.splitlines() if ("main.py" in l or "python" in l) and "ps aux" not in l]
        print("\n".join(matches) if matches else "nothing running")
    except Exception:
        print("nothing running")
    print("---")
    try:
        ss = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True).stdout
        hits = [l for l in ss.splitlines() if ":8188" in l]
        print("\n".join(hits) if hits else "nothing on 8188")
    except Exception:
        print("nothing on 8188")
    print("---")
    nv = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    print(nv.stdout.strip() if nv.returncode == 0 else "nvidia-smi not available")
    print("---")
    df = subprocess.run(["df", "-h", "/workspace"], capture_output=True, text=True)
    print(df.stdout.strip())
    print("---")
    cn = subprocess.run(["ls", str(COMFYUI_DIR / "custom_nodes")], capture_output=True, text=True)
    print(cn.stdout.strip())
    print("===========================================")


def kill_stale():
    subprocess.run(["fuser", "-k", "8188/tcp"], stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-9", "-f", "main.py --listen"], stderr=subprocess.DEVNULL)
    import time
    time.sleep(2)
    ss = subprocess.run(["ss", "-ltn"], capture_output=True, text=True).stdout
    if any(":8188 " in l for l in ss.splitlines()):
        print("WARNING: something is still listening on 8188 - check 'ps aux | grep main.py' and kill it manually")


def update_comfyui():
    run(["git", "-C", str(COMFYUI_DIR), "fetch", "origin"], check=True)
    subprocess.run(
        ["git", "-C", str(COMFYUI_DIR), "branch", "--set-upstream-to=origin/master", "master"],
        stderr=subprocess.DEVNULL,
    )
    run(["git", "-C", str(COMFYUI_DIR), "reset", "--hard", "origin/master"], check=True)


def install_deps():
    run([str(VENV_PIP), "install", "-r", str(COMFYUI_DIR / "requirements.txt"), "-q"], check=True)
    run([str(VENV_PIP), "install", "-U", "comfy-kitchen", "-q"], check=True)
    run([str(VENV_PIP), "install", "-U", "--pre", "comfyui-manager", "-q"], check=True)
    run([str(VENV_PIP), "install", "sageattention", "-q"], check=True)  # required for --use-sage-attention


def install_custom_nodes():
    custom_nodes = COMFYUI_DIR / "custom_nodes"
    extender_dir = custom_nodes / "ComfyUI_MiniMax_H3_Extender"
    if not extender_dir.is_dir():
        run(["git", "clone", "https://github.com/tritant/ComfyUI_MiniMax_H3_Extender.git"],
            cwd=custom_nodes, check=True)
    req = extender_dir / "requirements.txt"
    if req.is_file():
        run([str(VENV_PIP), "install", "-r", str(req), "-q"], check=True)
    # TODO: confirm repo URLs for "Spectrum" and "Prompt Builder" custom nodes, add clone calls here


def download_one(url: str, dest_subdir: str) -> bool:
    dest = BASE / dest_subdir
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["wget", "-c", *WGET_RETRY, "--show-progress",
           "--header", f"Authorization: Bearer {HF_TOKEN}", "-P", str(dest), url]
    result = run(cmd)
    return result.returncode == 0


def download_models(concurrency: int) -> bool:
    BASE.mkdir(parents=True, exist_ok=True)
    failed = False
    with ThreadPoolExecutor(max_workers=concurrency) as pool:  # sliding-window pool keeps exactly `concurrency` running
        futures = {pool.submit(download_one, url, sub): url for url, sub in DOWNLOADS}
        for future in as_completed(futures):
            if not future.result():
                failed = True
    return failed


def launch_comfyui():
    print("=== SETUP.PY DONE - models + custom nodes + lora ready. Launching ComfyUI now (Step C). ===")
    os.chdir(COMFYUI_DIR)
    os.execv(str(VENV_PY), [
        str(VENV_PY), "main.py",
        "--listen", "0.0.0.0", "--port", "8188",
        "--enable-cors-header", "--enable-manager", "--use-sage-attention",
    ])


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--auto-stop-worker":
        sys.exit(auto_stop_worker(sys.argv[2]))
    if len(sys.argv) == 3 and sys.argv[1] == "--auto-stop-worker-check":
        sys.exit(auto_stop_worker_check(sys.argv[2]))

    global HF_TOKEN
    HF_TOKEN = get_and_validate_hf_token()
    offer_pod_auto_stop()

    if len(sys.argv) > 1 and re.fullmatch(r"\d+", sys.argv[1]):
        concurrency = int(sys.argv[1])  # optional 1st arg overrides the prompt, e.g. `python3 setup.py 3`
    else:
        answer = input("How many downloads at a time? (HF rate-limits bigger bursts - default 2): ").strip()
        concurrency = int(answer) if re.fullmatch(r"\d+", answer) else 2

    os.chdir(COMFYUI_DIR)
    snapshot()
    kill_stale()
    update_comfyui()
    install_deps()
    install_custom_nodes()

    failed = download_models(concurrency)
    if failed:
        print("WARNING: at least one download above failed (often a 403 on the gated FLUX.2 repos)")
        print(GATED_REPOS_NOTE)
        print("=== SETUP.PY DONE WITH ERRORS - not auto-launching ComfyUI, fix the download above first, then run setup.py again ===")
        sys.exit(1)

    launch_comfyui()


if __name__ == "__main__":
    main()
