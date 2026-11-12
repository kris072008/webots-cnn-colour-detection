#!/bin/zsh
# Record the demo videos (real-time speed, 1280x720) with the demo_recorder
# supervisor, which adds captions from the robot's own log.
# Usage, from the repository root: cnn/results/videos/record_videos.sh
# The Python that Webots uses needs a TFLite runtime (ai-edge-litert or tensorflow).
set -eu
OUT=${0:A:h}
WEBOTS=${WEBOTS:-/Applications/Webots.app/Contents/MacOS/webots}
cd "${0:A:h}/../../../webots_project"

# macOS stops drawing a Webots window that is covered or in the background, which
# leaves black frames in the video. Each video is checked and re-recorded if needed;
# leave the Webots window visible while this runs.
PYTHON=${PYTHON:-python3}
black_frames() {  # prints the share of near-black frames in a video
  "$PYTHON" -I -c "import cv2, sys
c = cv2.VideoCapture(sys.argv[1]); n = int(c.get(7)); bad = 0
for k in range(0, n, 15):
    c.set(1, k); ok, f = c.read(); bad += (not ok) or f.mean() < 10
print(round(bad / max(1, len(range(0, n, 15))), 3))" "$1"
}

record() {  # name world seed seconds robot-args
  local name=$1 share
  for attempt in 1 2 3; do
    record_once "$@"
    if [[ ! -s "$OUT/$name.mp4" ]]; then
      echo "  $name: no video written - see $OUT/$name.log ($attempt/3)" >&2; continue
    fi
    share=$(black_frames "$OUT/$name.mp4") || share=""
    if [[ $share == 0 || $share == 0.0 ]]; then echo "  $name: OK"; return 0; fi
    echo "  $name: ${share:-unreadable} share of frames black (window hidden?) - recording again ($attempt/3)"
  done
  echo "  $name: FAILED after 3 tries" >&2
  FAILED=1
}

record_once() {  # name world seed seconds robot-args
  local name=$1 world=$2 seed=$3 seconds=$4 robot_args=$5
  local tmp="worlds/_video_$name.wbt"
  sed -e "s|  controller \"block_randomiser\"|  controller \"demo_recorder\"\n  controllerArgs [ \"--seed=$seed\" \"--seconds=$seconds\" \"--out=$OUT/$name.mp4\" ]|" \
      -e "s|  controller \"colour_detector\"|  controller \"colour_detector\"\n  controllerArgs [ $robot_args ]|" \
      "worlds/$world" > "$tmp"
  : > controllers/colour_detector/images/encounters.csv
  "$WEBOTS" --batch --mode=realtime --stdout --stderr "$tmp" > "$OUT/$name.log" 2>&1 || true
  rm -f "$tmp"
  grep -E "\[recorder\]" "$OUT/$name.log" | tail -2
}

FAILED=0
mkdir -p controllers/colour_detector/images
record demo_colour_blocks colour_blocks.wbt 1 50 ""
record distractors_hsv_only colour_blocks_distractors.wbt 1 62 "\"--shape-check=off\""
record distractors_shape_check colour_blocks_distractors.wbt 1 62 "\"--shape-check=on\""
exit $FAILED
