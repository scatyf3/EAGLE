#!/usr/bin/env bash

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"

usage() {
  cat <<'EOF'
Usage:
  sbatch_ssh.sh --reservation <name> --account <account> [options]

Required:
  --reservation <name>        Slurm reservation name (example: yf3005-h100)
  --account <account>         Slurm account (example: torch_pr_111_tandon_advanced)

Optional:
  --partition <name>          Slurm partition
  --node <name>               Fixed node (nodelist), example: gh015
  --gpus <N>                  GPUs requested (default: 1)
  --gpu-type <type>           GPU type in gres (default: h100)
  --cpus <N>                  CPUs per task (default: 4)
  --mem <size>                Memory (default: 64G)
  --time <HH:MM:SS>           Job walltime (default: 00:30:00)
  --hold-seconds <N>          sleep duration inside hold job (default: 1800)
  --job-name <name>           Slurm job name (default: ssh_hold)
  --poll-interval <sec>       Poll interval for queue checks (default: 3)
  --wait-timeout <sec>        Max wait for R state, 0 means no timeout (default: 300)
  --cancel-on-exit            Cancel hold job when SSH session exits
  --dry-run                   Print sbatch command only
  -h, --help                  Show this message

Examples:
  sbatch_ssh.sh \
    --reservation yf3005-h100 \
    --account torch_pr_111_tandon_advanced \
    --node gh015

  sbatch_ssh.sh \
    --reservation yf3005-h100 \
    --account torch_pr_111_tandon_advanced \
    --partition h100_tandon \
    --gpus 1 --cpus 4 --mem 64G --time 01:00:00 \
    --cancel-on-exit
EOF
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: missing required command '$1'" >&2
    exit 1
  fi
}

RESERVATION=""
ACCOUNT=""
PARTITION=""
NODE=""
GPUS="1"
GPU_TYPE="h100"
CPUS="4"
MEM="64G"
TIME_LIMIT="00:30:00"
HOLD_SECONDS="1800"
JOB_NAME="ssh_hold"
POLL_INTERVAL="3"
WAIT_TIMEOUT="300"
CANCEL_ON_EXIT="0"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reservation)
      RESERVATION="$2"
      shift 2
      ;;
    --account)
      ACCOUNT="$2"
      shift 2
      ;;
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    --node)
      NODE="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --gpu-type)
      GPU_TYPE="$2"
      shift 2
      ;;
    --cpus)
      CPUS="$2"
      shift 2
      ;;
    --mem)
      MEM="$2"
      shift 2
      ;;
    --time)
      TIME_LIMIT="$2"
      shift 2
      ;;
    --hold-seconds)
      HOLD_SECONDS="$2"
      shift 2
      ;;
    --job-name)
      JOB_NAME="$2"
      shift 2
      ;;
    --poll-interval)
      POLL_INTERVAL="$2"
      shift 2
      ;;
    --wait-timeout)
      WAIT_TIMEOUT="$2"
      shift 2
      ;;
    --cancel-on-exit)
      CANCEL_ON_EXIT="1"
      shift
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument '$1'" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$RESERVATION" || -z "$ACCOUNT" ]]; then
  echo "Error: --reservation and --account are required." >&2
  usage
  exit 1
fi

need_cmd sbatch
need_cmd squeue
need_cmd ssh

SBATCH_ARGS=(
  "--account=$ACCOUNT"
  "--reservation=$RESERVATION"
  "--ntasks=1"
  "--cpus-per-task=$CPUS"
  "--mem=$MEM"
  "--gres=gpu:$GPU_TYPE:$GPUS"
  "--time=$TIME_LIMIT"
  "--job-name=$JOB_NAME"
  "--parsable"
  "--wrap=sleep ${HOLD_SECONDS}"
)

if [[ -n "$PARTITION" ]]; then
  SBATCH_ARGS+=("--partition=$PARTITION")
fi

if [[ -n "$NODE" ]]; then
  SBATCH_ARGS+=("--nodelist=$NODE")
fi

echo "[$SCRIPT_NAME] sbatch args: ${SBATCH_ARGS[*]}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[$SCRIPT_NAME] dry run: sbatch ${SBATCH_ARGS[*]}"
  exit 0
fi

JOB_ID="$(sbatch "${SBATCH_ARGS[@]}")"
echo "[$SCRIPT_NAME] submitted job id: $JOB_ID"

cleanup() {
  if [[ "$CANCEL_ON_EXIT" == "1" ]]; then
    echo "[$SCRIPT_NAME] canceling job $JOB_ID"
    scancel "$JOB_ID" || true
  fi
}
trap cleanup EXIT

start_ts="$(date +%s)"

while true; do
  state="$(squeue -h -j "$JOB_ID" -o "%T" || true)"
  node_name="$(squeue -h -j "$JOB_ID" -o "%N" || true)"

  if [[ -z "$state" ]]; then
    echo "[$SCRIPT_NAME] job $JOB_ID no longer in queue." >&2
    exit 1
  fi

  echo "[$SCRIPT_NAME] job $JOB_ID state=$state node=$node_name"

  if [[ "$state" == "RUNNING" ]]; then
    target_node="$node_name"
    break
  fi

  if [[ "$WAIT_TIMEOUT" != "0" ]]; then
    now_ts="$(date +%s)"
    elapsed="$((now_ts - start_ts))"
    if (( elapsed >= WAIT_TIMEOUT )); then
      echo "[$SCRIPT_NAME] timeout waiting for RUNNING (>${WAIT_TIMEOUT}s)." >&2
      exit 1
    fi
  fi

  sleep "$POLL_INTERVAL"
done

echo "[$SCRIPT_NAME] entering node: $target_node"
ssh "$target_node"
