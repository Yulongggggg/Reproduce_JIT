#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
git submodule update --init --recursive
if [[ ! -x .env/bin/python ]]; then
  conda create -y -p "$PWD/.env" python=3.10 pip
fi
.env/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
.env/bin/python -m pip install -r requirements.txt
if [[ ! -d vendor/torch-fidelity ]]; then
  echo 'Initialize the pinned torch-fidelity submodule first.' >&2
  exit 1
fi
.env/bin/python -m pip install --no-deps -e vendor/torch-fidelity
.env/bin/python -m pip check
.env/bin/python -m pip freeze > reports/environment-lock.txt
