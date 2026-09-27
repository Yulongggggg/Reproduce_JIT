#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Serialize the independently queued four/eight-GPU publishers.
exec 9>.git/jit-report-publish.lock
flock -w 60 9
.env/bin/python scripts/report.py
git add reports/
if ! git diff --cached --quiet -- reports/; then
  git -c user.name='Yulongggggg' -c user.email='Yulongggggg@users.noreply.github.com' commit --only reports/ -m 'Update JiT reproduction evidence and measured status'
fi
# Refuse non-fast-forward updates; preserve any concurrent user changes.
git push origin HEAD:main
