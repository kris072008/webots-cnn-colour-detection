#!/bin/zsh
# End-to-end test of the shape CNN on the robot (report Section 4.8).
#
# Usage, from the repository root:
#   cnn/results/webots_runs/distractors/run_distractor_test.sh SEED on|off [SECONDS]
#
# Runs the unchanged colour_detector robot in worlds/colour_blocks_distractors.wbt
# (3 blocks + a red ball, a green cylinder and a blue ball, layout from SEED),
# with the demo_recorder supervisor scoring each inspection. "on" is the
# robot as submitted (it rejects objects the shape CNN says are not blocks);
# "off" ignores the shape CNN (the HSV-only behaviour) for comparison. Saves
# seed<SEED>_<on|off>_encounters.csv and _ground_truth.csv next to this script.
# The Python that Webots uses needs a TFLite runtime (ai-edge-litert or tensorflow).
set -eu
SEED=$1; MODE=$2; SECONDS_SIM=${3:-150}
[[ $MODE == on || $MODE == off ]] || { echo "MODE must be on or off" >&2; exit 2; }
OUT=${0:A:h}
WEBOTS=${WEBOTS:-/Applications/Webots.app/Contents/MacOS/webots}
cd "${0:A:h}/../../../../webots_project"
IMAGES=controllers/colour_detector/images
mkdir -p "$IMAGES"
: > "$IMAGES/encounters.csv"   # start this run's log empty
rm -f "$IMAGES/ground_truth.csv"  # never copy a previous run's ground truth
NAME="seed${SEED}_${MODE}"
WORLD="worlds/_distractor_$NAME.wbt"
sed -e "s|  controller \"block_randomiser\"|  controller \"demo_recorder\"\n  controllerArgs [ \"--seed=$SEED\" \"--seconds=$SECONDS_SIM\" ]|" \
    -e "s|  controller \"colour_detector\"|  controller \"colour_detector\"\n  controllerArgs [ \"--shape-check=$MODE\" ]|" \
    worlds/colour_blocks_distractors.wbt > "$WORLD"
grep -q 'controller "demo_recorder"' "$WORLD" || { echo "world edit failed: no demo_recorder" >&2; exit 1; }
"$WEBOTS" --batch --mode=fast --minimize --no-rendering --stdout --stderr "$WORLD" > "$OUT/$NAME.log" 2>&1 || true
rm -f "$WORLD"
[[ -s "$IMAGES/ground_truth.csv" ]] || { echo "no ground truth written - see $OUT/$NAME.log" >&2; exit 1; }
cp "$IMAGES/encounters.csv" "$OUT/${NAME}_encounters.csv"
cp "$IMAGES/ground_truth.csv" "$OUT/${NAME}_ground_truth.csv"
grep -E "\[recorder\]|WARNING" "$OUT/$NAME.log" | head -5
