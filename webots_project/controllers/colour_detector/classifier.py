"""CIFAR-10 CNN inference for the robot, run off the control loop.

Optimisation proposal, implemented
----------------------------------
The control loop must finish every 32 ms step on time, or the motors and
obstacle avoidance stall. CNN inference time depends on the host: well under
1 ms for this small int8 model on a modern laptop (see the notebook,
Section 9), but tens of ms on a slow robot CPU, plus Python start-up costs on
the first call. So this module never lets inference touch the loop's timing:

1. It prefers a quantised TensorFlow Lite model (``cifar10_cnn_int8.tflite``,
   exported by the notebook). The model is much smaller, and integer
   arithmetic suits embedded CPUs and accelerators such as the Coral Edge TPU.
2. It runs inference on a background worker thread. ``submit()`` returns at
   once. The queue holds one frame: if the CNN is busy, the older frame is
   dropped rather than letting a backlog build up (latest-frame-wins).
3. It is only called on events (block reached, key C pressed, start-up), never
   on every frame. Cheap HSV colour detection is the pre-filter that decides
   when the CNN is worth running.

If no model file or runtime is installed, the classifier says why once and the
robot keeps working with colour detection only. Inference errors come back as
results with ``error`` set, and after repeated failures the CNN switches itself
off and says so.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2 as cv
import numpy as np

CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck")
INPUT_SIZE = 32  # CIFAR-10 images are 32x32x3
MAX_CONSECUTIVE_FAILURES = 3

Backend = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Prediction:
    tag: str  # what the frame was (e.g. "red_block_003"), for the log
    label: str = ""
    confidence: float = 0.0  # softmax probability of ``label``, 0..1
    top3: tuple[tuple[str, float], ...] = ()
    latency_ms: float = 0.0  # preprocessing + forward pass
    error: str = ""  # non-empty if inference failed


def preprocess(bgr_square: np.ndarray) -> np.ndarray:
    """BGR crop -> (1, 32, 32, 3) float32 RGB in [0, 1], the training format.

    The notebook's ``load_for_cnn`` uses this exact resize (OpenCV INTER_AREA),
    so the robot and the notebook feed the CNN identical pixels.
    """
    rgb = cv.cvtColor(bgr_square, cv.COLOR_BGR2RGB)
    small = cv.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv.INTER_AREA)
    return (small.astype(np.float32) / 255.0)[np.newaxis, ...]


def softmax(logits: np.ndarray) -> np.ndarray:
    """Logits -> probabilities. The model's last layer outputs raw logits
    (training used from_logits=True). Subtracting the max avoids overflow."""
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum()


class _TFLiteBackend:
    def __init__(self, model_path: Path) -> None:
        interpreter_cls = _import_tflite_interpreter()
        self._interp = interpreter_cls(model_path=str(model_path), num_threads=1)
        self._interp.allocate_tensors()
        self._inp = self._interp.get_input_details()[0]
        self._out = self._interp.get_output_details()[0]
        _check_shapes(tuple(self._inp["shape"]), tuple(self._out["shape"]))

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        x = batch
        if self._inp["dtype"] != np.float32:  # integer-quantised input
            scale, zero = self._inp["quantization"]
            info = np.iinfo(self._inp["dtype"])
            x = np.clip(np.round(batch / scale + zero), info.min, info.max).astype(self._inp["dtype"])
        self._interp.set_tensor(self._inp["index"], x)
        self._interp.invoke()
        y = self._interp.get_tensor(self._out["index"])[0]
        if self._out["dtype"] != np.float32:  # dequantise the logits
            scale, zero = self._out["quantization"]
            y = (y.astype(np.float32) - zero) * scale
        return y


class _KerasBackend:
    def __init__(self, model_path: Path) -> None:
        import tensorflow as tf  # heavy import, only when needed

        self._model = tf.keras.models.load_model(model_path)
        _check_shapes(tuple(self._model.input_shape), tuple(self._model.output_shape))

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        return np.asarray(self._model(batch, training=False))[0]


def _check_shapes(inp: tuple, out: tuple) -> None:
    if tuple(inp[1:]) != (INPUT_SIZE, INPUT_SIZE, 3) or out[-1] != len(CLASS_NAMES):
        raise ValueError(f"model expects input {inp} / gives output {out}; "
                         f"need (*, {INPUT_SIZE}, {INPUT_SIZE}, 3) -> (*, {len(CLASS_NAMES)})")


def _import_tflite_interpreter() -> type:
    """Find a TFLite interpreter: LiteRT, tflite-runtime, or full TensorFlow."""
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter
    except ImportError:
        pass
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("no TFLite runtime found (tried ai_edge_litert, tflite_runtime, "
                          "tensorflow) - pip install ai-edge-litert") from None
    return tf.lite.Interpreter


def load_backend(model_dir: Path) -> tuple[Backend | None, str]:
    """Return (backend, description). backend is None if nothing could load.

    Tries the int8 model first, then float32 TFLite, then Keras. If a preferred
    model failed and a fallback loaded, the description says so loudly.
    """
    candidates = [
        (model_dir / "cifar10_cnn_int8.tflite", _TFLiteBackend),
        (model_dir / "cifar10_cnn.tflite", _TFLiteBackend),
        (model_dir / "cifar10_cnn.keras", _KerasBackend),
    ]
    errors = []
    for path, backend_cls in candidates:
        if not path.is_file():
            continue
        try:
            backend = backend_cls(path)
        except Exception as exc:  # any load failure: record it and try the next file
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        description = f"{path.name} ({backend_cls.__name__.strip('_')})"
        if errors:
            description += "  [FALLBACK - earlier models failed: " + " | ".join(errors) + "]"
        return backend, description
    if errors:
        return None, "could not load a model -> " + " | ".join(errors)
    return None, (f"no model file in {model_dir} (run the notebook and copy "
                  "cifar10_cnn_int8.tflite there)")


class AsyncClassifier:
    """Background-thread CNN. ``submit`` never blocks the control loop.

    Only the main thread calls submit/poll; only the worker increments
    ``failures``. Queues handle all hand-over between the two threads.
    """

    def __init__(self, model_dir: Path, backend: Backend | None = None) -> None:
        if backend is None:
            self.backend, self.description = load_backend(model_dir)
        else:  # injected backend (used by the unit tests)
            self.backend, self.description = backend, "injected backend"
        self.enabled = self.backend is not None
        self._requests: queue.Queue[tuple[str, np.ndarray]] = queue.Queue(maxsize=1)
        self._results: queue.Queue[Prediction] = queue.Queue()
        self.dropped_frames = 0
        self.failures = 0
        if self.enabled:
            threading.Thread(target=self._worker, name="cnn-worker", daemon=True).start()

    def submit(self, tag: str, bgr_square: np.ndarray) -> bool:
        """Queue a frame for classification. Returns False if the CNN is off."""
        if not self.enabled or bgr_square.size == 0:
            return False
        try:
            self._requests.put_nowait((tag, bgr_square.copy()))
        except queue.Full:
            # Worker still busy: replace the waiting frame with the newer one.
            try:
                self._requests.get_nowait()
                self.dropped_frames += 1
            except queue.Empty:
                pass
            self._requests.put_nowait((tag, bgr_square.copy()))
        return True

    def poll(self) -> list[Prediction]:
        """All results finished since the last call (non-blocking)."""
        done = []
        while True:
            try:
                done.append(self._results.get_nowait())
            except queue.Empty:
                return done

    def _classify(self, tag: str, frame: np.ndarray) -> Prediction:
        start = time.perf_counter()
        probs = softmax(self.backend(preprocess(frame)))
        if probs.shape != (len(CLASS_NAMES),) or not np.isfinite(probs).all():
            raise ValueError(f"model returned invalid output {probs!r}")
        latency = (time.perf_counter() - start) * 1000.0
        order = np.argsort(probs)[::-1][:3]
        return Prediction(
            tag=tag,
            label=CLASS_NAMES[order[0]],
            confidence=float(probs[order[0]]),
            top3=tuple((CLASS_NAMES[i], float(probs[i])) for i in order),
            latency_ms=latency,
        )

    def _worker(self) -> None:
        consecutive_failures = 0
        while self.enabled:
            tag, frame = self._requests.get()
            try:
                result = self._classify(tag, frame)
                consecutive_failures = 0
            except Exception as exc:  # never let the thread die silently
                self.failures += 1
                consecutive_failures += 1
                result = Prediction(tag=tag, error=f"{type(exc).__name__}: {exc}")
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    self.enabled = False
                    self.description += f"  [DISABLED after {consecutive_failures} failed inferences]"
            self._results.put(result)
