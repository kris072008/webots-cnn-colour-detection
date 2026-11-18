# Demo run sheet

## Main demo: the CNN object tour (`worlds/cnn_objects.wbt`)

| Step | Do | Point out |
|---|---|---|
| 1 | Before class: open `cnn_objects.wbt` from this folder (File → Open World) on Wi-Fi and wait for the models to load | Console: `[init] CNN ON` and, if set up, `Grad-CAM ON` |
| 2 | Let it run until the checklist (top right of the 3D view) shows 2–3 objects, then ⏸ | The yellow ball marks the robot; `[lidar] … 8 objects found`: nobody told it where they are |
| 3 | Explain: blob → block by HSV; otherwise 3 photos → int8 CNN votes; under 60% or disagreeing views → "unsure" | `[cnn] truck view 1/3 …` lines, `[0.3 ms, off the control loop]` |
| 4 | Point at the top-left window | Live Grad-CAM: where the CNN looked, e.g. the truck's wheel and cab |
| 5 | Press ⏩ | `Done: 7 of 8 named correctly`; the paper boat is "airplane" even on water |

If `Could not find the controller directory` appears, the world was opened from a copy outside this folder: reopen it from `webots_project/worlds/`.

## Colour-block search (`worlds/colour_blocks.wbt`)

The demo takes about 3 minutes. Rehearse it once the day before.

## Before class (10 minutes)

1. Open Webots R2025a and check the console shows no errors from `File → Open World… → worlds/colour_blocks.wbt`.
2. Check that `controllers/colour_detector/models/cifar10_cnn_int8.tflite` exists. The console should say `CNN ON` at start-up. If it says `CNN OFF`, the demo still works with colour detection only.
3. Delete `controllers/colour_detector/images/` so the demo starts with a clean capture folder.
4. Open `controllers/colour_detector/images/` in Finder, next to Webots. New photos will appear there live.
5. Fix the layout so the run is repeatable: in the scene tree, set the `block randomiser` Robot node's `controllerArgs` to `"1"`. With seed 1, all three blocks were found in 42.5 s of simulated time during testing.
6. Do one full rehearsal, then press ⏮ (reset) to get back to the start.

## Live demo

| Step | Do | Point out |
|---|---|---|
| 1 | Press ▶ at normal speed (1×) | The supervisor prints a random layout of three blocks, 0.3 × 0.3 × 0.1 m each |
| 2 | Watch the console `[fsm]` lines | SEARCH and SCAN until a block is seen, then APPROACH with proportional steering |
| 3 | First `[found]` line | The robot stops, saves raw, annotated and 32×32 CNN-input images, and the CNN result appears a moment later on the background thread |
| 4 | Show the `*_annotated.png` in Finder | The HSV mask box: grass, sky and walls are ignored |
| 5 | Let it find the next colours (speed up with ⏩ if needed) | It skips colours already found and goes looking for the others |
| 6 | `round 1 complete` and the SUMMARY | Blocks found, vision cost of about 0.08 ms per frame, CNN predictions and latency |
| 7 | Optional: open `worlds/cnn_camera_test.wbt` | The robot faces a dog (a CIFAR-10 class) and the CNN classifies its camera frame |
| 8 | Optional: click the 3D view and press C | Manual capture and classify |
| 9 | Optional, the strongest moment: open `worlds/colour_blocks_distractors.wbt` | A red ball, a green cylinder and a blue ball share the blocks' colours. Watch the console: `[found] BLUE object ... - checking its shape`, then `[shape] BLUE object is a ball (..%), not a block - not counted, keep searching`. The robot goes on to find the real blue block. |
| 10 | Optional before/after: in the e-puck's `controllerArgs` add `--shape-check=off` and run again | The HSV-only robot counts the ball as "the blue block" and stops looking. That is the colour-only weakness the shape CNN fixes. |

## Talking points

- **Why HSV rather than RGB:** one hue range per colour works in both light and shade, which the lecture's RGB filter could not do.
- **Why the CNN mislabels the blocks:** a coloured box is not one of the 10 CIFAR-10 classes, and the model has no "unknown" output. That motivates the proposal to train a detector on Webots images.
- **Optimisation:** inference runs off the control loop, only on events, on an int8 model. The robot never stalls.
- **The proposal, implemented:** the shape CNN was trained only on Webots images that a supervisor labelled automatically. It decides whether the coloured thing is really a block, so the robot is no longer fooled by a ball of the same colour.

## If something goes wrong

| Problem | Quick fix |
|---|---|
| `ModuleNotFoundError: cv2` | `python3 -m pip install opencv-python` for the Python shown in the console |
| `CNN OFF` | Carry on without it: colour detection is the required demo. Explain the CNN with the notebook's results. |
| The robot circles for a long time | Press ⏩, or reset and set seed `1` |
| Webots is slow on the projector laptop | Turn off rendering of the 3D view (`View → Rendering`); the console and saved images still show everything |
