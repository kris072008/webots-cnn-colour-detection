# Applied Robotics and AI Project: Part 2

Module 5FTC2068, University of Hertfordshire / PSB Academy.

| Folder | What it contains |
|---|---|
| [`cnn/`](cnn/) | `cifar10_cnn.ipynb`: CIFAR-10 CNN training, evaluation, Webots-camera predictions and TFLite quantisation benchmark. `own_photos.ipynb`: the trained CNN on 18 of my own photos (`my_photos/`) |
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

**Does the CNN slow the robot down?** With 50 ms added to every inference to emulate a slow embedded CPU, running the CNN inside the control loop on every frame missed the 32 ms deadline in 50% of steps (worst step 135 ms). The deployed design, with a background thread and event-triggered inference, missed none ([`timing/`](cnn/results/webots_runs/timing/)).

Figures, model files and the model card are in [`cnn/results/`](cnn/results/). Webots run logs and captures are in [`cnn/results/webots_runs/`](cnn/results/webots_runs/).

## Quick start

```bash
cd webots_project
python3 -m pip install opencv-python numpy pytest
python3 -m pytest tests -q          # 39 unit tests
```

Open `webots_project/worlds/colour_blocks.wbt` in Webots and press ▶.

## AI use

This project was developed with substantial help from an AI assistant (Claude, Anthropic). The AI wrote and tested much of the code and ran the experiments. This is declared in the project report.
