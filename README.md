# Applied Robotics and AI Project: Part 2

Module 5FTC2068, University of Hertfordshire / PSB Academy.

| Folder | What it contains |
|---|---|
| [`cnn/`](cnn/) | `cifar10_cnn.ipynb`: CIFAR-10 CNN training, evaluation, Webots-camera predictions and TFLite quantisation benchmark. `own_photos.ipynb`: the trained CNN on 18 of my own photos (`my_photos/`). `gradcam.ipynb`: what the CNN looks at. `webots_shapes.ipynb`: a CNN trained on auto-labelled Webots images (`webots_shapes/`). `webots_animals.ipynb`: the CNN on all 8 Webots animals seen by the robot (`webots_animals/`). `webots_object_tour.ipynb`: the CNN on the robot's object-tour photos (`webots_object_tour/`). **`run_webots_detection_colab.ipynb`: self-contained, runs in Google Colab with Run all: the robot's int8 CNN and Grad-CAM on 17 Webots images, plus upload-your-own-image** |
| [`cnn/results/videos/`](cnn/results/videos/) | Demo videos: the normal search, and the distractor world without and with the shape check |
| [`webots_project/`](webots_project/) | Webots R2025a package: e-puck colour-block search with OpenCV HSV detection and off-loop CNN inference. See its [README](webots_project/README.md) and [DEMO](webots_project/DEMO.md). |

## Results (CPU training run, seed 42)

| Model | Validation | Test (10,000 images) | Parameters |
|---|---|---|---|
| Lecture baseline + SGD (10 epochs) | 59.62% | 60.61% | 122,570 |
| Lecture baseline + Adam (10 epochs) | 67.12% | 66.90% | 122,570 |
| **Improved CNN + Adam** (80 epochs) | **87.18%** | **86.47%** ± 0.67% (95% CI) | 306,154 |

| Deployment variant | Size | Median latency (M4 CPU) | Test accuracy |
|---|---|---|---|
| Keras float32 | 3.8 MB | 4.85 ms | 86.47% |
| TFLite float32 | 1226 kB | 0.12 ms | 86.47% |
| TFLite full int8 (used on the robot) | 328 kB | 0.09 ms | 86.57% |

**My own photos** ([`cnn/own_photos.ipynb`](cnn/own_photos.ipynb)): on 18 phone photos of vehicles, every side view was classified correctly (5 of 5), but only 2 of 7 front and rear views, a viewpoint bias from CIFAR-10. The int8 robot model agreed with the full model on all 18.

**A CNN trained on the robot's own world** ([`cnn/webots_shapes.ipynb`](cnn/webots_shapes.ipynb)): a supervisor labels Webots camera crops automatically, and a CNN learns to tell blocks from same-coloured balls and cylinders. On 25 unseen layouts it scores 99.2% (3 seeds), against 94.5% for the best hand-made features; the deployed int8 model scores 99.4%. On the robot, in a world with distractors, colour detection alone counted 28 of 28 distractors as blocks; with the shape check the robot counted 0 of 26 and kept searching ([`distractors/`](cnn/results/webots_runs/distractors/)).

**Every Webots animal, through the robot's camera** ([`cnn/webots_animals.ipynb`](cnn/webots_animals.ipynb)): the robot photographed all eight animals in the Webots library from 24 directions. Deer and horse were right in 43 of 48 views, cat and dog in none (both mostly "horse"). Replacing only the background with grey made the dog right in 12 of 24 views and dropped the deer from 21 to 4, so the CNN reads the scenery as much as the animal.

**A CNN object tour** ([`cnn/webots_object_tour.ipynb`](cnn/webots_object_tour.ipynb), world `cnn_objects.wbt`): the robot finds three blocks and five Webots objects by itself with a 360° lidar, names them (HSV for blocks, the int8 CNN with a three-view vote and an "unsure" answer for the rest) and ticks them off a checklist drawn on the 3D view. All 8 found, 7 named correctly; a paper boat was called "airplane" even standing on water. Grad-CAM on these photos and on a bird photo of my own: [`cnn/gradcam_tour_bird.ipynb`](cnn/gradcam_tour_bird.ipynb).

**Does the CNN slow the robot down?** With 50 ms added to every inference to emulate a slow embedded CPU, running the CNN inside the control loop on every frame missed the 32 ms deadline in 50% of steps (worst step 135 ms). The deployed design, with a background thread and event-triggered inference, missed none ([`timing/`](cnn/results/webots_runs/timing/)).

Figures, model files and the model card are in [`cnn/results/`](cnn/results/). Webots run logs and captures are in [`cnn/results/webots_runs/`](cnn/results/webots_runs/).

## Quick start

```bash
cd webots_project
python3 -m pip install opencv-python numpy pytest
python3 -m pytest tests -q          # 75 unit tests
```

Open `webots_project/worlds/colour_blocks.wbt` in Webots and press ▶.
