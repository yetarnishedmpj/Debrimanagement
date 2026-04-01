#!/usr/bin/env bash
set -Eeuo pipefail

# Production-friendly bootstrap script for the orbital debris MARL prototype.
#
# The defaults are safe, but the knobs below are meant to be easy to change
# while debugging deployment issues or adapting the stack to a different machine.
#
# Useful overrides:
#   PYTHON_BIN="py -3" ./setup_and_test.sh
#   INSTALL_MODE=phased ./setup_and_test.sh
#   TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121" ./setup_and_test.sh
#   SKIP_TESTS=1 ./setup_and_test.sh
#   RECREATE_VENV=1 ./setup_and_test.sh
#   EXTRA_PIP_ARGS="--timeout 180 --retries 10" ./setup_and_test.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-${ROOT_DIR}/requirements.txt}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
INSTALL_MODE="${INSTALL_MODE:-phased}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
SKIP_TESTS="${SKIP_TESTS:-0}"
RECREATE_VENV="${RECREATE_VENV:-0}"
EXTRA_PIP_ARGS="${EXTRA_PIP_ARGS:-}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
TEST_TARGET="${TEST_TARGET:-${ROOT_DIR}/tests}"
PYTEST_FLAGS="${PYTEST_FLAGS:--q}"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/bootstrap_$(date +%Y%m%d_%H%M%S).log"

log() {
  printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "${LOG_FILE}"
}

fail() {
  log "ERROR: $*"
  exit 1
}

run_cmd() {
  log "+ $*"
  "$@" 2>&1 | tee -a "${LOG_FILE}"
}

run_shell_cmd() {
  local cmd="$1"
  log "+ ${cmd}"
  bash -lc "${cmd}" 2>&1 | tee -a "${LOG_FILE}"
}

resolve_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    # shellcheck disable=SC2206
    PYTHON_CMD=( ${PYTHON_BIN} )
    return
  fi

  if command -v python >/dev/null 2>&1; then
    PYTHON_CMD=( python )
    return
  fi

  if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD=( python3 )
    return
  fi

  if command -v py >/dev/null 2>&1; then
    PYTHON_CMD=( py -3 )
    return
  fi

  fail "No Python interpreter found. Set PYTHON_BIN before rerunning."
}

resolve_venv_python() {
  if [[ -f "${VENV_DIR}/Scripts/python.exe" ]]; then
    VENV_PYTHON="${VENV_DIR}/Scripts/python.exe"
    return
  fi

  if [[ -f "${VENV_DIR}/bin/python" ]]; then
    VENV_PYTHON="${VENV_DIR}/bin/python"
    return
  fi

  fail "Virtual environment created, but its Python executable was not found."
}

create_or_recreate_venv() {
  if [[ "${RECREATE_VENV}" == "1" && -d "${VENV_DIR}" ]]; then
    log "Removing existing virtual environment at ${VENV_DIR}"
    rm -rf "${VENV_DIR}"
  fi

  if [[ ! -d "${VENV_DIR}" ]]; then
    log "Creating virtual environment at ${VENV_DIR}"
    run_cmd "${PYTHON_CMD[@]}" -m venv "${VENV_DIR}"
  else
    log "Reusing existing virtual environment at ${VENV_DIR}"
  fi

  resolve_venv_python
}

upgrade_packaging_tools() {
  run_cmd "${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel ${EXTRA_PIP_ARGS}
}

install_requirements_one_shot() {
  local pip_args=( "${VENV_PYTHON}" -m pip install -r "${REQUIREMENTS_FILE}" )
  if [[ -n "${EXTRA_PIP_ARGS}" ]]; then
    # shellcheck disable=SC2206
    pip_args+=( ${EXTRA_PIP_ARGS} )
  fi
  run_cmd "${pip_args[@]}"
}

install_requirements_phased() {
  local common_args=( )
  if [[ -n "${EXTRA_PIP_ARGS}" ]]; then
    # shellcheck disable=SC2206
    common_args+=( ${EXTRA_PIP_ARGS} )
  fi

  # Install the lightweight scientific stack first. If something fails here,
  # the log will usually point to the exact wheel or compiler issue.
  run_cmd "${VENV_PYTHON}" -m pip install \
    numpy==1.26.4 dm-tree==0.1.8 requests==2.32.5 matplotlib==3.10.8 plotly==5.24.1 streamlit==1.39.0 pytest==8.4.2 \
    "${common_args[@]}"

  run_cmd "${VENV_PYTHON}" -m pip install gymnasium==0.28.1 "${common_args[@]}"

  # Torch is separated because production environments often need a custom wheel
  # index (CPU, CUDA 11.8, CUDA 12.1, etc.).
  if [[ -n "${TORCH_INDEX_URL}" ]]; then
    run_cmd "${VENV_PYTHON}" -m pip install \
      --index-url "${TORCH_INDEX_URL}" \
      torch==2.5.1 \
      "${common_args[@]}"
  else
    run_cmd "${VENV_PYTHON}" -m pip install torch==2.5.1 "${common_args[@]}"
  fi

  run_cmd "${VENV_PYTHON}" -m pip install "ray[rllib]==2.40.0" "${common_args[@]}"
}

run_validation() {
  run_cmd "${VENV_PYTHON}" -m compileall "${ROOT_DIR}"
  run_cmd "${VENV_PYTHON}" -m pip check

  if [[ "${SKIP_TESTS}" == "1" ]]; then
    log "SKIP_TESTS=1, skipping pytest."
    return
  fi

  # shellcheck disable=SC2206
  local pytest_flags=( ${PYTEST_FLAGS} )
  run_cmd "${VENV_PYTHON}" -m pytest "${pytest_flags[@]}" "${TEST_TARGET}"
}

main() {
  resolve_python
  log "Using bootstrap log ${LOG_FILE}"
  log "Resolved host Python command: ${PYTHON_CMD[*]}"
  log "Install mode: ${INSTALL_MODE}"

  create_or_recreate_venv
  log "Resolved venv Python: ${VENV_PYTHON}"

  if [[ "${SKIP_INSTALL}" != "1" ]]; then
    upgrade_packaging_tools

    case "${INSTALL_MODE}" in
      phased)
        install_requirements_phased
        ;;
      one-shot)
        install_requirements_one_shot
        ;;
      *)
        fail "Unsupported INSTALL_MODE='${INSTALL_MODE}'. Use 'phased' or 'one-shot'."
        ;;
    esac
  else
    log "SKIP_INSTALL=1, skipping dependency installation."
  fi

  run_validation
  log "Bootstrap finished successfully."
}

main "$@"

