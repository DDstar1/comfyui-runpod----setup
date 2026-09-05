#!/usr/bin/env python3
"""Python port of setup.sh - cold-start ComfyUI on a fresh RunPod pod.
Usage: python3 setup.py [concurrency]   (concurrency defaults to 2)
"""
import os
import re
import shutil
import signal
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

TMUX_SESSION = "comfy-setup"


def timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def stage(name: str):
    print(f"\n{'=' * 72}", flush=True)
    print(f"[{timestamp()}] {name}", flush=True)
    print(f"{'=' * 72}", flush=True)


def ensure_tmux():
    """
    Make tmux the first real setup operation.

    The outer process creates the persistent session and immediately attaches
    to it. The inner process sees TMUX and continues normally. If the session
    already exists, attach to it rather than creating a second setup process.
    """
    if os.environ.get("TMUX"):
        return

    script = str(Path(__file__).resolve())
    args = [sys.executable, script, *sys.argv[1:]]

    existing = subprocess.run(
        ["tmux", "has-session", "-t", TMUX_SESSION],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if existing.returncode == 0:
        print(f"[{timestamp()}] tmux session '{TMUX_SESSION}' already exists.")
        print("Attaching to existing setup session...", flush=True)
        os.execvp("tmux", ["tmux", "attach-session", "-t", TMUX_SESSION])

    print(f"[{timestamp()}] Creating persistent tmux session '{TMUX_SESSION}'...", flush=True)

    subprocess.run(
        ["tmux", "new-session", "-d", "-s", TMUX_SESSION, *args],
        check=True,
    )

    os.execvp("tmux", ["tmux", "attach-session", "-t", TMUX_SESSION])

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
    print(f"[{timestamp()}] + {' '.join(display_cmd)}", flush=True)
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
    answer = input("Permanently terminate this RunPod after 2 hours? [y/N]: ").strip().lower()
    if answer not in ("y", "yes"):
        print("Automatic pod termination not scheduled.")
        return

    pod_id = os.environ.get("RUNPOD_POD_ID", "").strip()
    if not pod_id:
        print("WARNING: RUNPOD_POD_ID is not set; skipping automatic termination.", file=sys.stderr)
        return
    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    while True:
        if api_key:
            try:
                runpod_pod_request(pod_id, api_key, action="verify")
                break
            except RuntimeError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)

        if not sys.stdin.isatty():
            print("WARNING: cannot request another API key without an interactive terminal; skipping automatic termination.", file=sys.stderr)
            return

        choice = input("Press Enter to provide another RunPod API key, or type 'skip' to continue without termination: ").strip().lower()
        if choice in ("s", "skip"):
            print("Automatic pod termination skipped; continuing setup.")
            return
        api_key = masked_input("Paste your RunPod API key: ").strip()
        if not api_key:
            print("No API key entered; try again or choose 'skip'.", file=sys.stderr)

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
    print(f"Permanent termination scheduled for pod {pod_id} in 2 hours.")


def runpod_pod_request(pod_id: str, api_key: str, action: str):
    if action not in ("verify", "terminate"):
        raise ValueError(f"unknown RunPod pod action: {action}")

    encoded_pod_id = urllib.parse.quote(pod_id, safe="")
    url = "https://rest.runpod.io/v1/pods"
    if action == "terminate":
        url += f"/{encoded_pod_id}"
    request = urllib.request.Request(
        url,
        method="GET" if action == "verify" else "DELETE",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
            payload = json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise RuntimeError("RunPod rejected the API key or it lacks pod permissions.") from exc
        if exc.code == 404:
            raise RuntimeError(f"RunPod could not find pod {pod_id} under this API-key account.") from exc
        raise RuntimeError(f"RunPod pod request failed (HTTP {exc.code}).") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not connect to the RunPod API: {exc}") from exc

    if action == "verify":
        pods = payload if isinstance(payload, list) else payload.get("pods", [])
        if not any(isinstance(pod, dict) and pod.get("id") == pod_id for pod in pods):
            raise RuntimeError(f"RunPod could not find pod {pod_id} under this API-key account.")

    return payload


def auto_stop_worker(pod_id: str):
    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key:
        print("Automatic termination failed: RUNPOD_API_KEY is missing.", file=sys.stderr)
        return 1
    time.sleep(2 * 60 * 60)
    try:
        runpod_pod_request(pod_id, api_key, action="terminate")
    except RuntimeError as exc:
        print(f"Automatic termination failed: {exc}", file=sys.stderr)
        return 1
    print(f"Permanent termination requested successfully for pod {pod_id}.")
    return 0


def auto_stop_worker_check(pod_id: str):
    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key:
        print("ERROR: RUNPOD_API_KEY is missing.", file=sys.stderr)
        return 1
    try:
        runpod_pod_request(pod_id, api_key, action="verify")
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
    fuser = shutil.which("fuser")
    ss = shutil.which("ss")
    pkill = shutil.which("pkill")

    if fuser:
        subprocess.run([fuser, "-k", "8188/tcp"], stderr=subprocess.DEVNULL)
    elif ss:
        listeners = subprocess.run([ss, "-ltnp"], capture_output=True, text=True).stdout
        for line in listeners.splitlines():
            if ":8188" not in line:
                continue
            for pid in set(re.findall(r"pid=(\d+)", line)):
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass

    if pkill:
        subprocess.run([pkill, "-9", "-f", "main.py --listen"], stderr=subprocess.DEVNULL)
    else:
        ps = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True).stdout
        for line in ps.splitlines():
            if "main.py --listen" not in line:
                continue
            fields = line.strip().split(maxsplit=1)
            if fields and fields[0].isdigit():
                try:
                    os.kill(int(fields[0]), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass

    time.sleep(2)
    listeners = subprocess.run([ss, "-ltn"], capture_output=True, text=True).stdout if ss else ""
    if any(":8188 " in line for line in listeners.splitlines()):
        print("WARNING: something is still listening on 8188 - check 'ps aux | grep main.py' and kill it manually")


def update_comfyui():
    run(["git", "-C", str(COMFYUI_DIR), "fetch", "origin"], check=True)
    subprocess.run(
        ["git", "-C", str(COMFYUI_DIR), "branch", "--set-upstream-to=origin/master", "master"],
        stderr=subprocess.DEVNULL,
    )
    run(["git", "-C", str(COMFYUI_DIR), "reset", "--hard", "origin/master"], check=True)


def install_deps():
    stage("Installing ComfyUI Python requirements")
    run([
        str(VENV_PIP),
        "install",
        "-r",
        str(COMFYUI_DIR / "requirements.txt"),
    ], check=True)

    stage("Installing comfy-kitchen")
    run([
        str(VENV_PIP),
        "install",
        "-U",
        "comfy-kitchen",
    ], check=True)

    stage("Installing ComfyUI Manager")
    run([
        str(VENV_PIP),
        "install",
        "-U",
        "--pre",
        "comfyui-manager",
    ], check=True)

    stage("Installing SageAttention")
    run([
        str(VENV_PIP),
        "install",
        "sageattention",
    ], check=True)  # required for --use-sage-attention


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
    print(f"[{timestamp()}] Starting download: {url}", flush=True)
    result = run(cmd)
    if result.returncode == 0:
        print(f"[{timestamp()}] Download completed: {url}", flush=True)
    else:
        print(f"[{timestamp()}] Download FAILED: {url}", flush=True)
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
    print(
        f"[{timestamp()}] === SETUP.PY DONE - models + custom nodes + lora ready. "
        "Launching ComfyUI now (Step C). ===",
        flush=True,
    )
    os.chdir(COMFYUI_DIR)
    os.execv(str(VENV_PY), [
        str(VENV_PY), "main.py",
        "--listen", "0.0.0.0", "--port", "8188",
        "--enable-cors-header", "--enable-manager", "--use-sage-attention",
    ])


def main():
    # IMPORTANT: this must happen before all normal setup work.
    # Worker modes must bypass tmux because they are deliberately detached.
    if len(sys.argv) == 3 and sys.argv[1] == "--auto-stop-worker":
        sys.exit(auto_stop_worker(sys.argv[2]))
    if len(sys.argv) == 3 and sys.argv[1] == "--auto-stop-worker-check":
        sys.exit(auto_stop_worker_check(sys.argv[2]))

    ensure_tmux()

    stage("ComfyUI RunPod setup started")

    global HF_TOKEN
    stage("Hugging Face authentication")
    HF_TOKEN = get_and_validate_hf_token()

    stage("RunPod auto-stop configuration")
    offer_pod_auto_stop()

    if len(sys.argv) > 1 and re.fullmatch(r"\d+", sys.argv[1]):
        concurrency = int(sys.argv[1])
        print(f"[{timestamp()}] Model download concurrency: {concurrency}", flush=True)
    else:
        answer = input(
            "How many downloads at a time? "
            "(HF rate-limits bigger bursts - default 2): "
        ).strip()
        concurrency = int(answer) if re.fullmatch(r"\d+", answer) else 2
        print(f"[{timestamp()}] Model download concurrency: {concurrency}", flush=True)

    os.chdir(COMFYUI_DIR)

    stage("System snapshot")
    snapshot()

    stage("Cleaning stale ComfyUI processes")
    kill_stale()

    stage("Updating ComfyUI")
    update_comfyui()

    install_deps()

    stage("Installing custom nodes")
    install_custom_nodes()

    stage(f"Downloading models with concurrency={concurrency}")
    failed = download_models(concurrency)

    if failed:
        print(
            "WARNING: at least one download above failed "
            "(often a 403 on the gated FLUX.2 repos)",
            flush=True,
        )
        print(GATED_REPOS_NOTE, flush=True)
        print(
            "=== SETUP.PY DONE WITH ERRORS - not auto-launching ComfyUI, "
            "fix the download above first, then run setup.py again ===",
            flush=True,
        )
        sys.exit(1)

    stage("Launching ComfyUI inside tmux")
    launch_comfyui()


if __name__ == "__main__":
    main()
