#!/bin/zsh
# Reproduce the inference-timing experiment (report Section 4.7).
#
# Usage, from the repository root:
#   cnn/results/webots_runs/timing/run_timing_experiment.sh NAME "CONTROLLER ARGS" [fast|realtime]
# Example:
#   cnn/results/webots_runs/timing/run_timing_experiment.sh F_sync_frame_50 \
#       "--cnn-mode=sync --cnn-trigger=every-frame --emulate-ms=50 --stop-after=90"
#
# It copies worlds/colour_blocks.wbt with the given e-puck controllerArgs and
# block-layout seed 1, runs Webots headless until the controller writes
# images/step_timing.json, and saves that file as NAME.json next to this script.
# The Python that Webots uses needs a TFLite runtime (ai-edge-litert or tensorflow).
set -eu
NAME=$1; ARGS=$2; MODE=${3:-fast}
OUT=${0:A:h}
WEBOTS=${WEBOTS:-/Applications/Webots.app/Contents/MacOS/webots}
cd "${0:A:h}/../../../../webots_project"

WORLD="worlds/_timing_${NAME}.wbt"
ARGS_WBT=$(echo "$ARGS" | awk '{for (i = 1; i <= NF; i++) printf "\"%s\" ", $i}')
sed -e "s|  controller \"colour_detector\"|  controller \"colour_detector\"\n  controllerArgs [ ${ARGS_WBT}]|" \
    -e "s|  controller \"block_randomiser\"|  controller \"block_randomiser\"\n  controllerArgs [ \"1\" ]|" \
    worlds/colour_blocks.wbt > "$WORLD"
REPORT=controllers/colour_detector/images/step_timing.json
rm -f "$REPORT"

"$WEBOTS" --batch --mode="$MODE" --minimize --no-rendering --stdout --stderr "$WORLD" > "$OUT/$NAME.log" 2>&1 &
PID=$!
for _ in {1..400}; do [[ -f "$REPORT" ]] && break; sleep 1; done
sleep 1; kill $PID 2>/dev/null || true
rm -f "$WORLD"
if [[ -f "$REPORT" ]]; then
  cp "$REPORT" "$OUT/$NAME.json"
  grep "\[timing\]" "$OUT/$NAME.log"
else
  echo "no timing report - see $OUT/$NAME.log" >&2; exit 1
fi
