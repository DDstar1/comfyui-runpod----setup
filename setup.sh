#!/usr/bin/env bash
set -euo pipefail

# Keep the Bash entry point behavior identical to setup.py by using the Python
# implementation as the single source of truth. Resolve the script directory so
# this works regardless of the caller's current working directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/setup.py" "$@"
