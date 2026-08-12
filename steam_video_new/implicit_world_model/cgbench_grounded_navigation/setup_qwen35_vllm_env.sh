#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/fs/gamma-projects/vlm-robot/conda/envs/swift/bin/python3.10}"
VLLM_ROOT="${VLLM_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills/.venv-qwen35-vllm}"
VLLM_VERSION="${VLLM_VERSION:-0.25.1}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python interpreter is not executable: ${PYTHON}" >&2
  exit 2
fi

if [[ ! -d "${VLLM_ROOT}" ]]; then
  "${PYTHON}" -m venv --copies "${VLLM_ROOT}"
fi

"${VLLM_ROOT}/bin/python" -m pip install --upgrade pip
"${VLLM_ROOT}/bin/python" -m pip install "vllm==${VLLM_VERSION}"
"${VLLM_ROOT}/bin/python" - <<'PY'
import vllm
print(f"vllm={vllm.__version__}")
PY

printf 'vllm_root=%s\n' "${VLLM_ROOT}"
