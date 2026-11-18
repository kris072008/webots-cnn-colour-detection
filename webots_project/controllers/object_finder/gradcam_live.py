"""Live Grad-CAM for the object tour, on a background thread.

The robot names objects with the int8 TensorFlow Lite model, which has no
gradients. Grad-CAM therefore uses the full Keras model of the same network
(cnn/results/models/cifar10_cnn.keras), with the same preprocessing. It only
runs once per object, after the vote, on a worker thread, so the control loop
never waits for it. If TensorFlow or the Keras model is missing, ``available``
is False and the tour runs exactly as before, without heatmaps.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path

import cv2 as cv
import numpy as np

CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
LAST_CONV = "activation_5"  # last conv block output (8x8x128), as in cnn/gradcam.ipynb
_HERE = Path(__file__).resolve()
# The Keras model ships next to the int8 one in a submission package; in the full repository it lives in cnn/results/models/.
_CANDIDATES = [_HERE.parents[1] / "colour_detector" / "models" / "cifar10_cnn.keras",
               _HERE.parents[3] / "cnn" / "results" / "models" / "cifar10_cnn.keras"]
MODEL = next((m for m in _CANDIDATES if m.is_file()), _CANDIDATES[-1])


def overlay(rgb32: np.ndarray, cam: np.ndarray, size: int) -> np.ndarray:
    """32x32 RGB input in [0,1] + heatmap in [0,1] -> size x size BGR image."""
    big = cv.resize((rgb32 * 255).astype(np.uint8), (size, size), interpolation=cv.INTER_NEAREST)
    heat = cv.applyColorMap((cv.resize(cam, (size, size), interpolation=cv.INTER_CUBIC).clip(0, 1) * 255)
                           .astype(np.uint8), cv.COLORMAP_JET)
    return cv.addWeighted(cv.cvtColor(big, cv.COLOR_RGB2BGR), 0.55, heat, 0.45, 0)


class LiveGradCam:
    def __init__(self) -> None:
        self.available = False
        self.reason = ""
        self._jobs: queue.Queue = queue.Queue()
        self._done: queue.Queue = queue.Queue()
        try:
            import tensorflow as tf
            import keras
            model = keras.models.load_model(MODEL)
            self._grad_model = keras.Model(model.inputs, [model.get_layer(LAST_CONV).output, model.output])
            self._tf = tf
        except Exception as exc:  # no TensorFlow, or no model: run without heatmaps, and say why
            self.reason = f"{type(exc).__name__}: {exc}"
            return
        self.available = True
        threading.Thread(target=self._worker, name="gradcam", daemon=True).start()

    def submit(self, tag: str, bgr_square: np.ndarray) -> None:
        if self.available:
            self._jobs.put((tag, bgr_square.copy()))

    def poll(self) -> list[tuple[str, np.ndarray | None, str, float, str]]:
        out = []
        while True:
            try:
                out.append(self._done.get_nowait())
            except queue.Empty:
                return out

    def _compute(self, bgr_square: np.ndarray):
        tf = self._tf
        rgb = cv.cvtColor(bgr_square, cv.COLOR_BGR2RGB)
        x = cv.resize(rgb, (32, 32), interpolation=cv.INTER_AREA).astype(np.float32) / 255.0
        with tf.GradientTape() as tape:
            maps, logits = self._grad_model(x[None], training=False)
            cls = int(tf.argmax(logits[0]))
            score = logits[:, cls]
        grads = tape.gradient(score, maps)[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam = tf.nn.relu(tf.reduce_sum(maps[0] * weights, axis=-1)).numpy()
        cam = cv.resize(cam, (32, 32), interpolation=cv.INTER_CUBIC)
        cam = np.clip(cam, 0, None)
        cam = cam / cam.max() if cam.max() > 0 else cam
        prob = float(tf.nn.softmax(logits[0]).numpy()[cls])
        return x, cam, CLASS_NAMES[cls], prob

    def _worker(self) -> None:
        while True:
            tag, frame = self._jobs.get()
            try:
                x, cam, label, prob = self._compute(frame)
                self._done.put((tag, overlay(x, cam, 256), label, prob, ""))
            except Exception as exc:  # report, never kill the thread silently
                self._done.put((tag, None, "", 0.0, f"{type(exc).__name__}: {exc}"))
