#!/bin/bash
# Run a command detached from the terminal, appending output to a log, and re-launch it if it dies
# (e.g. killed by another user's process or a GPU guard). The command must itself be resumable
# (train_nav.py resumes from last.ckpt; convert_navdp.py rewrites its .part file).
#
# Usage: nav/run_resilient.sh <name> <max_restarts> <command...>
#   -> log: $LEWM_LOG_DIR/<name>.log, pid file: $LEWM_LOG_DIR/<name>.pid
set -u
NAME=$1; MAX=$2; shift 2
LOG_DIR=${LEWM_LOG_DIR:-/home/nyuair/yuxuan/lewm-nav/logs}
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/$NAME.log"

(
  for ((i = 0; i <= MAX; i++)); do
    echo "=== [$(date '+%F %T')] attempt $i: $*" >> "$LOG"
    "$@" >> "$LOG" 2>&1
    rc=$?
    echo "=== [$(date '+%F %T')] exit code $rc" >> "$LOG"
    [ $rc -eq 0 ] && { echo "=== FINISHED" >> "$LOG"; exit 0; }
    sleep 60
  done
  echo "=== GAVE UP after $MAX restarts" >> "$LOG"
) < /dev/null > /dev/null 2>&1 &
echo $! > "$LOG_DIR/$NAME.pid"
echo "started $NAME (pid $!), log: $LOG"
