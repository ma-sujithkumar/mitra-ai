#!/usr/bin/env bash
#
# MITRA setup script: install backend (pip) and frontend (npm) dependencies.
#
# Resolves the repo root from the script location and the Python interpreter
# portably (config.ini [python] PYTHON => local venv => python3/python on PATH).
# ASCII only; status lines are prefixed with "=>".

set -euo pipefail

# Resolve the directory that contains this script, then the repo root one level up.
# This keeps the script invokable from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_INI_PATH="${REPO_ROOT}/config.ini"
REQUIREMENTS_PATH="${REPO_ROOT}/requirements.txt"
FRONTEND_DIR="${REPO_ROOT}/frontend"


read_python_from_config() {
  # Print the [python] PYTHON= value from config.ini, or nothing if unset/missing.
  # Uses awk to track the active section so we only read PYTHON under [python].
  if [ ! -f "${CONFIG_INI_PATH}" ]; then
    return 0
  fi
  awk '
    /^[[:space:]]*\[/ {
      section = $0
      sub(/^[[:space:]]*\[/, "", section)
      sub(/\][[:space:]]*$/, "", section)
      next
    }
    {
      if (tolower(section) == "python") {
        line = $0
        # Match a PYTHON key (case-insensitive) and emit its value.
        if (line ~ /^[[:space:]]*[Pp][Yy][Tt][Hh][Oo][Nn][[:space:]]*=/) {
          sub(/^[[:space:]]*[Pp][Yy][Tt][Hh][Oo][Nn][[:space:]]*=[[:space:]]*/, "", line)
          print line
        }
      }
    }
  ' "${CONFIG_INI_PATH}"
}


resolve_python() {
  # Portable resolution order: config.ini => repo venv => python3/python on PATH.
  local configured
  configured="$(read_python_from_config)"

  if [ -n "${configured}" ]; then
    # Bash expands "~" automatically in most contexts, but a value read out of
    # a config file via awk is plain text, so expand a leading "~" ourselves.
    case "${configured}" in
      "~"|"~/"*) configured="${HOME}${configured#\~}" ;;
    esac
    # Treat as a path relative to the repo root when it is not absolute.
    case "${configured}" in
      /*) candidate="${configured}" ;;
      *)  candidate="${REPO_ROOT}/${configured}" ;;
    esac
    if [ -x "${candidate}" ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
    # Otherwise assume it is a bare command available on PATH.
    printf '%s\n' "${configured}"
    return 0
  fi

  local venv_candidates=(
    "${REPO_ROOT}/.venv/bin/python"
    "${REPO_ROOT}/venv/bin/python"
  )
  for candidate in "${venv_candidates[@]}"; do
    if [ -x "${candidate}" ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  if command -v python3 >/dev/null 2>&1; then
    printf '%s\n' "python3"
    return 0
  fi
  printf '%s\n' "python"
}


main() {
  echo "=> MITRA setup starting"
  echo "=> repo root: ${REPO_ROOT}"

  local python_bin
  python_bin="$(resolve_python)"
  echo "=> using python interpreter: ${python_bin}"

  # Backend dependencies: only attempt pip install when requirements.txt exists.
  if [ -f "${REQUIREMENTS_PATH}" ]; then
    echo "=> installing backend dependencies from requirements.txt"
    "${python_bin}" -m pip install -r "${REQUIREMENTS_PATH}"
    echo "=> backend dependencies installed"
  else
    echo "=> WARNING: no requirements.txt at repo root, skipping pip install"
  fi

  # Frontend dependencies via npm inside the frontend directory.
  if [ -d "${FRONTEND_DIR}" ]; then
    echo "=> installing frontend dependencies (npm install) in ${FRONTEND_DIR}"
    ( cd "${FRONTEND_DIR}" && npm install )
    echo "=> frontend dependencies installed"
  else
    echo "=> WARNING: no frontend directory at ${FRONTEND_DIR}, skipping npm install"
  fi

  echo "=> MITRA setup complete"
}


main "$@"
