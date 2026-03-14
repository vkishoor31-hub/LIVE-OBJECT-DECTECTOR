"""
Galaxy Scan - Standalone Live Animal Detection (Console Mode)
Runs YOLO inference in a background thread; draws HUD on the live webcam window.
Auto-loads fine-tuned model if models/animal_detector.pt exists.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Dict, List, Optional

# ── CV2 ───────────────────────────────────────────────────────────────────────
_cv2: Any = None
try:
    import cv2 as _cv2_mod  # type: ignore[import]
    _cv2 = _cv2_mod
except ImportError:
    pass

# ── pyttsx3 ───────────────────────────────────────────────────────────────────
_pyttsx3: Any = None
try:
    import pyttsx3 as _pyttsx3_mod  # type: ignore[import]
    _pyttsx3 = _pyttsx3_mod
except ImportError:
    pass

# ── YOLO ──────────────────────────────────────────────────────────────────────
_YOLO: Any = None
try:
    from ultralytics import YOLO as _YOLO_cls  # type: ignore[import]
    _YOLO = _YOLO_cls
except ImportError:
    pass

# ── Model loader ──────────────────────────────────────────────────────────────
_CUSTOM_MODEL = Path("models") / "animal_detector.pt"
USE_CUSTOM_CLASSES: bool = _CUSTOM_MODEL.exists()

model: Any = None
if _YOLO is not None:
    try:
        if _CUSTOM_MODEL.exists():
            model = _YOLO(str(_CUSTOM_MODEL))
            print(f"[Model] ✅ Fine-tuned model loaded: {_CUSTOM_MODEL}")
        else:
            model = _YOLO("yolov8m.pt")
            print("[Model] YOLOv8m loaded (run train_model.py for higher accuracy)")
    except Exception as _exc:
        print(f"[Model] Error: {_exc}")
        try:
            model = _YOLO("yolov8s.pt")
            print("[Model] Fallback → YOLOv8s")
        except Exception:
            model = None

# COCO class IDs: bird=14, cat=15, dog=16, horse=17, sheep=18, cow=19,
#                 elephant=20, bear=21, zebra=22, giraffe=23
COCO_ANIMAL_IDS: List[int] = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
CUSTOM_MAX_CLS: int = 9  # fine-tuned model uses indices 0-9

# ── Shared state ──────────────────────────────────────────────────────────────
_lock = Lock()
_current_frame: Any = None

_det: Dict[str, Any] = {
    "label": "",
    "conf": 0.0,
    "pixel_area": 0,
    "box": [50, 50, 150, 150],
}

_last_spoken: Dict[str, float] = {}
SPEAK_DELAY: float = 1.5


# ── Voice ─────────────────────────────────────────────────────────────────────
def _speak(text: str) -> None:
    if _pyttsx3 is None:
        return
    try:
        engine: Any = _pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass


# ── Inference worker ──────────────────────────────────────────────────────────
def _inference_worker() -> None:
    global _current_frame
    while True:
        frame_copy: Any = None
        with _lock:
            if _current_frame is not None:
                frame_copy = _current_frame.copy()

        if frame_copy is not None and model is not None:
            try:
                results: Any = model.predict(
                    frame_copy, stream=True, verbose=False,
                    conf=0.30, iou=0.45, agnostic_nms=True,
                )
            except Exception as exc:
                print(f"[Inference] {exc}")
                time.sleep(0.01)
                continue

            top_label: str = ""
            top_conf: float = 0.0
            top_area: int = 0
            top_box: List[int] = [50, 50, 150, 150]
            found: bool = False

            for r in results:
                if not hasattr(r, "boxes") or r.boxes is None:
                    continue
                if not hasattr(model, "names") or model.names is None:
                    continue
                for b in r.boxes:
                    try:
                        cls: int = int(b.cls[0].item())
                        if USE_CUSTOM_CLASSES:
                            if cls < 0 or cls > CUSTOM_MAX_CLS:
                                continue
                        else:
                            if cls not in COCO_ANIMAL_IDS:
                                continue
                        conf: float = float(b.conf[0].item())
                        if conf < 0.30:
                            continue
                        name: str = str(model.names[cls])
                        coords: Any = b.xyxy[0]
                        x1 = int(coords[0])
                        y1 = int(coords[1])
                        x2 = int(coords[2])
                        y2 = int(coords[3])
                        area: int = (x2 - x1) * (y2 - y1)
                        if not found or conf > top_conf:
                            top_label, top_conf = name, conf
                            top_area = area
                            top_box = [x1, y1, x2, y2]
                            found = True
                    except Exception:
                        continue

            now: float = time.time()
            with _lock:
                if found:
                    _det["label"] = top_label
                    _det["conf"] = top_conf
                    _det["pixel_area"] = top_area
                    _det["box"] = top_box

                    last_t: float = float(_last_spoken.get(top_label, 0.0))
                    if now - last_t > SPEAK_DELAY:
                        _last_spoken[top_label] = now
                        pct: int = int(top_conf * 100)
                        msg: str = f"{top_label.upper()} detected with {pct} percent accuracy"
                        Thread(target=_speak, args=(msg,), daemon=True).start()
                else:
                    _det["label"] = ""
                    _det["conf"] = 0.0
                    _det["pixel_area"] = 0
                    _det["box"] = [50, 50, 150, 150]

        time.sleep(0.01)


Thread(target=_inference_worker, daemon=True).start()


# ── Main capture loop ─────────────────────────────────────────────────────────
def main() -> None:
    global _current_frame

    if _cv2 is None:
        print("[ERROR] OpenCV not installed.")
        return

    cap: Any = _cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[CRITICAL] Camera not found.")
        return

    while True:
        ok: bool
        frame: Any
        ok, frame = cap.read()  # type: ignore[misc]
        if not ok:
            cap.release()
            time.sleep(1)
            cap = _cv2.VideoCapture(0)
            continue

        snapshot: Dict[str, Any] = {}
        with _lock:
            _current_frame = frame.copy()
            snapshot = _det.copy()

        # ── HUD ───────────────────────────────────────────────────────────
        label_s: str = str(snapshot.get("label", ""))
        if label_s:
            cf: float = float(snapshot.get("conf", 0.0))
            acc: float = int(cf * 1000) / 10.0
            raw_box: Any = snapshot.get("box", [50, 50, 150, 150])
            if isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
                try:
                    bx1 = int(float(raw_box[0]))
                    by1 = int(float(raw_box[1]))
                    bx2 = int(float(raw_box[2]))
                    by2 = int(float(raw_box[3]))
                    lbl: str = label_s.upper() if label_s else "DETECTING"
                    _cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
                    px: int = int(float(snapshot.get("pixel_area", 0)))
                    _cv2.putText(frame, f"{lbl} {acc}% | {px}px",
                        (bx1, by1 - 10), _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                except Exception:
                    pass

        _cv2.putText(frame, "GALAXY SCAN  //  CONSOLE MODE",
            (10, 35), _cv2.FONT_HERSHEY_SIMPLEX, 0.65, (168, 85, 247), 2)
        _cv2.imshow("Galaxy AI Object Detector", frame)

        if _cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    _cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
