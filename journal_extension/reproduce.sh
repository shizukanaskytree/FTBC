#!/usr/bin/env bash
set -euo pipefail
extension_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-python3}"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--solver-check" ) ]]; then
    echo 'Usage: bash journal_extension/reproduce.sh [--solver-check]' >&2
    exit 2
fi
"$python_bin" "$extension_dir/scripts/reproduce.py"
"$python_bin" "$extension_dir/scripts/summarize_control_sensitivity.py" --verify-frozen
if [[ $# -eq 1 ]]; then
    "$python_bin" "$extension_dir/scripts/check_solver.py"
fi
