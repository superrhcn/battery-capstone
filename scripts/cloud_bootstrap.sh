#!/usr/bin/env bash
# EC2 user-data entrypoint. Parent must replace both __CAPSTONE_*__ tokens.
# No AWS API, IAM role, SSH key, secret, or inbound port is required.
set -Eeuo pipefail
umask 077
BASE=/var/lib/battery-capstone
mkdir -p "$BASE"
exec > >(tee -a "$BASE/bootstrap.log" /dev/console) 2>&1

finish() {
    rc=$?
    trap - EXIT
    printf 'CAPSTONE_BOOTSTRAP_EXIT=%s\n' "$rc"
    printf 'CAPSTONE_EBS_OUTPUT_ROOT=%s\n' "$BASE"
    sync
    if [ "$rc" -eq 0 ]; then
        /usr/sbin/shutdown -h now
    else
        printf 'Failure recorded; power-off in one minute. Inspect bootstrap.log and batch outputs.\n'
        /usr/sbin/shutdown -h +1
    fi
    exit "$rc"
}
trap finish EXIT
# Independent of this shell, pip, the batch supervisor and solver processes.
systemd-run --unit=capstone-hard-stop --on-active=12h \
    --timer-property=AccuracySec=1s /usr/sbin/shutdown -h now

REPO_URL="${CAPSTONE_REPO_URL:-__CAPSTONE_REPO_URL__}"
REV="${CAPSTONE_GIT_COMMIT:-__CAPSTONE_GIT_COMMIT__}"
[[ "$REPO_URL" =~ ^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(\.git)?$ ]]
[[ "$REV" =~ ^[0-9a-f]{40}$ ]]
export GIT_TERMINAL_PROMPT=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
printf 'CAPSTONE_BOOTSTRAP_BEGIN commit=%s\n' "$REV"
# AL2023 keeps system Python3.9; install3.11 alongside it, never replace system Python.
# https://docs.aws.amazon.com/linux/al2023/ug/python.html
/usr/bin/timeout 1200 dnf install -y python3.11 python3.11-pip git libgomp
/usr/bin/timeout 300 git clone --no-checkout "$REPO_URL" "$BASE/repo"
/usr/bin/timeout 300 git -C "$BASE/repo" checkout --detach "$REV"
ACTUAL=$(git -C "$BASE/repo" rev-parse HEAD)
[ "$ACTUAL" = "$REV" ]
python3.11 -m venv "$BASE/venv"
/usr/bin/timeout 1200 "$BASE/venv/bin/python" -m pip install --only-binary=:all: \
    -r "$BASE/repo/requirements-simulation.txt"
"$BASE/venv/bin/python" -m pip freeze > "$BASE/pip-freeze.txt"
python3.11 --version
printf 'Linux Python patch version may differ from Windows3.11.9; Linux integration gate runs before100EFC.\n'
cd "$BASE/repo"
"$BASE/venv/bin/python" simulation/run_cloud_batch.py \
    --output-dir "$BASE/results" --batch-max-wall-s 39600
