# Applied Robotics and AI Project: Part 2

Module 5FTC2068, University of Hertfordshire / PSB Academy.

| Folder | What it contains |
|---|---|
| [`cnn/`](cnn/) | `cifar10_cnn.ipynb`: CIFAR-10 CNN training, evaluation, own-image and Webots-camera predictions, and TFLite quantisation benchmark |
| [`webots_project/`](webots_project/) | Webots R2025a package: e-puck colour-block search with OpenCV HSV detection and off-loop CNN inference. See its [README](webots_project/README.md) and [DEMO](webots_project/DEMO.md). |

## Quick start

```bash
cd webots_project
python3 -m pip install opencv-python numpy pytest
python3 -m pytest tests -q          # 23 unit tests
```

Open `webots_project/worlds/colour_blocks.wbt` in Webots and press ▶.

## AI use

This project was developed with substantial help from an AI assistant (Claude, Anthropic). The AI wrote and tested much of the code and ran the experiments. This is declared in the project report.
