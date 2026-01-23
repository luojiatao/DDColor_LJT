#!/usr/bin/env bash
# Minimal helper: activate a conda env in non-interactive shells.
# Usage: source "$(dirname "$0")/_activate_conda.sh" ddcolor

set -euo pipefail

ENV_NAME="${1:-ddcolor}"

if [[ "${CONDA_DEFAULT_ENV:-}" == "$ENV_NAME" ]]; then
  return 0
fi

_conda_sh=""

if command -v conda >/dev/null 2>&1; then
  _conda_base="$(conda info --base 2>/dev/null || true)"
  if [[ -n "$_conda_base" && -f "${_conda_base}/etc/profile.d/conda.sh" ]]; then
    _conda_sh="${_conda_base}/etc/profile.d/conda.sh"
  fi
fi

# Common locations when `conda` isn't on PATH
if [[ -z "$_conda_sh" ]]; then
  for p in "$HOME/anaconda3" "$HOME/miniconda3" "/opt/conda"; do
    if [[ -f "${p}/etc/profile.d/conda.sh" ]]; then
      _conda_sh="${p}/etc/profile.d/conda.sh"
      break
    fi
  done
fi

if [[ -z "$_conda_sh" ]]; then
  echo "[activate_conda] conda.sh not found; please ensure conda is installed/initialized." >&2
  echo "[activate_conda] expected: <conda_base>/etc/profile.d/conda.sh" >&2
  return 1
fi

# shellcheck disable=SC1090
source "$_conda_sh"
conda activate "$ENV_NAME"
