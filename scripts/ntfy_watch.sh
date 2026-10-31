#!/bin/bash
# Posts run events from the local logs to an ntfy.sh topic: every finished
# screen arm / training run, every stage change and every failure.
#   NTFY_TOPIC=<topic> setsid nohup bash scripts/ntfy_watch.sh > /dev/null 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
: "${NTFY_TOPIC:?set NTFY_TOPIC}"
LOGS=(logs/screen.log logs/screen2.log logs/overnight_data.log logs/v2_queue.log
      logs/biencoder_probe.log)
touch "${LOGS[@]}"

send() {  # $1 = title, $2 = message
  curl -s -m 20 -H "Title: $1" -d "$2" "https://ntfy.sh/$NTFY_TOPIC" \
    > /dev/null || true
}

tail -n 0 -F "${LOGS[@]}" 2>/dev/null |
  grep --line-buffered -E \
    '^==> |^=== |RESULT|^!!!|^DONE|^DECISION|Traceback|Error:|stage: ' |
  while IFS= read -r line; do
    case "$line" in
      "==> "*) src="${line#==> }"; src="${src% <==}"; continue ;;
    esac
    case "$line" in
      *Traceback*|*Error:*|'!!!'*) title="opensrma: FAILURE" ;;
      *RESULT*|DONE*|DECISION*) title="opensrma: run finished" ;;
      *) title="opensrma: progress" ;;
    esac
    send "$title" "[$(basename "${src:-log}")] ${line:0:900}"
  done
