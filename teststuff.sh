#!/usr/bin/env bash

run_with_timeout() {
  local cmd="$1"
  local secs="$2"

  CMD="$cmd" python3 - "$secs" <<'PY'
import os, subprocess, sys

cmd = os.environ["CMD"]
timeout = float(sys.argv[1])

try:
    # Inherit stdout/stderr so you see Python errors in the terminal
    completed = subprocess.run(cmd, shell=True, timeout=timeout)
    raise SystemExit(completed.returncode)
except subprocess.TimeoutExpired:
    raise SystemExit(124)
PY
}

CMDS=(
  "python3 backup_1.py"
  "python3 backup_2.py"
  "python3 backup_1_correct_training.py"
)

for cmd in "${CMDS[@]}"; do
  echo "Running: $cmd"

  if run_with_timeout "$cmd" 5; then
    echo "  -> OK (exited cleanly)"
  else
    status=$?
    if [ "$status" -eq 124 ]; then
      echo "  -> Timed out after 5 seconds"
    else
      echo "  -> Crashed / exit code $status"
    fi
  fi

  echo
done
