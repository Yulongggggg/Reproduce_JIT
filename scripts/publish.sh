#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.env/bin/python scripts/report.py
git add reports/
if ! git diff --cached --quiet; then
  git -c user.name='Yulongggggg' -c user.email='Yulongggggg@users.noreply.github.com' commit -m 'Update JiT reproduction evidence and measured status'
fi
# Refuse non-fast-forward updates; preserve any concurrent user changes.
git push origin HEAD:main
