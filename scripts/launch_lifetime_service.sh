#!/usr/bin/env bash
# Invoke from SSM/root after reviewing --plan from run_lifetime_cloud.py.
# This launches exactly one independent job; it is not enabled for every boot.
set -Eeuo pipefail
umask 077
if [ "${EUID}" -ne 0 ]; then
    printf 'Run this launcher as Linux root via SSM.\n' >&2
    exit 2
fi
if [ "$#" -eq 0 ]; then
    printf 'Required: --run-root PATH and either --import-100efc-dir PATH or --resume.\n' >&2
    exit 2
fi
for argument in "$@"; do
    case "$argument" in
        --plan|--help|-h)
            printf 'For read-only inspection invoke run_lifetime_cloud.py directly; this launcher powers off EC2 on exit.\n' >&2
            exit 2;;
    esac
done
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${CAPSTONE_REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
PYTHON_PATH=${CAPSTONE_PYTHON:-/var/lib/battery-capstone/venv/bin/python}
SERVICE_UNIT=${CAPSTONE_SERVICE_UNIT:-capstone-lifetime}
[[ "$SERVICE_UNIT" =~ ^[a-zA-Z0-9_-]+$ ]]
[ -f "$REPO_ROOT/scripts/run_lifetime_cloud.py" ]
[ -x "$PYTHON_PATH" ]
# Type=exec reports failure to exec immediately. No --wait: SSM/chat can close.
# ExecStopPost also stops EC2 if the supervisor itself crashes before its finally.
systemd-run --unit="$SERVICE_UNIT" \
    --property=Type=exec --property=Restart=no --property=KillMode=control-group \
    --property=TimeoutStopSec=60s \
    --property='ExecStopPost=/usr/sbin/shutdown -h +1' --collect \
    "$PYTHON_PATH" "$REPO_ROOT/scripts/run_lifetime_cloud.py" \
    --repo-root "$REPO_ROOT" --python "$PYTHON_PATH" "$@"
printf 'Inspect: systemctl status %s.service --no-pager\n' "$SERVICE_UNIT"
printf 'Check run-root/cloud_status.json before closing the setup session.\n'
