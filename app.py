"""
Galaxy Scan - Live Animal Detection Web Server
Flask backend with auto-loading fine-tuned YOLO model, voice alerts, SMS.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Dict, List, Optional, Tuple

# ── Flask ─────────────────────────────────────────────────────────────────────
from flask import (  # type: ignore[import]
    Flask, Response, flash, jsonify, redirect,
    render_template, request, session, url_for,
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash

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

# ── Ultralytics YOLO ──────────────────────────────────────────────────────────
_YOLO: Any = None
try:
    from ultralytics import YOLO as _YOLO_cls  # type: ignore[import]
    _YOLO = _YOLO_cls
except ImportError:
    pass

# ── Twilio ────────────────────────────────────────────────────────────────────
_TwilioClient: Any = None
try:
    from twilio.rest import Client as _TC  # type: ignore[import]
    _TwilioClient = _TC
except ImportError:
    pass

# ── dotenv ────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv  # type: ignore[import]
    load_dotenv()
except ImportError:
    pass

# ── App & Twilio setup ────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = str(os.getenv("SECRET_KEY", "galaxy_secret_2024"))
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login_page"

# ── Database Model ────────────────────────────────────────────────────────────
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

with app.app_context():
    db.create_all()

_SID: Optional[str] = os.getenv("TWILIO_ACCOUNT_SID")
_AUTH: Optional[str] = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE: Optional[str] = os.getenv("TWILIO_PHONE_NUMBER")

twilio_client: Any = None
if _SID and _AUTH and _TwilioClient is not None:
    try:
        twilio_client = _TwilioClient(_SID, _AUTH)
    except Exception:
        twilio_client = None

TWILIO_READY: bool = twilio_client is not None
if TWILIO_READY:
    print("[Twilio] ✅ Client initialized successfully")
else:
    print("[Twilio] ⚠️ Client not initialized (check .env credentials)")

# ── YOLO model (smart loader) ─────────────────────────────────────────────────
# Priority: 1) fine-tuned animal_detector.pt  2) yolov8m.pt (auto-download)
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
_CUSTOM_MODEL = MODELS_DIR / "animal_detector.pt"
USE_CUSTOM_CLASSES: bool = _CUSTOM_MODEL.exists()

model: Any = None
if _YOLO is not None:
    try:
        if _CUSTOM_MODEL.exists():
            model = _YOLO(str(_CUSTOM_MODEL))
            print(f"[Model] ✅ Fine-tuned model loaded: {_CUSTOM_MODEL}")
        else:
            # Prefer yolov8s.pt if it exists locally to avoid downloading yolov8m.pt
            if Path("yolov8s.pt").exists():
                model = _YOLO("yolov8s.pt")
                print("[Model] ✅ Local YOLOv8s loaded")
            else:
                model = _YOLO("yolov8m.pt")   # downloads if not present
                print("[Model] YOLOv8m loaded (run train_model.py for better accuracy)")
    except Exception as _exc:
        print(f"[Model] Error: {_exc}")
        try:
            model = _YOLO("yolov8n.pt")
            print("[Model] Fallback → YOLOv8n")
        except Exception:
            model = None

# COCO class IDs for 10 animal categories
# bird=14, cat=15, dog=16, horse=17, sheep=18, cow=19
# elephant=20, bear=21, zebra=22, giraffe=23
COCO_ANIMAL_IDS: List[int] = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

# Fine-tuned model uses class indices 0-9
CUSTOM_MAX_CLS: int = 9

# ── Inference Config & State ────────────────────────────────────────────────
_INF_CONFIG = {
    "conf": 0.30,
    "iou": 0.45,
    "precise": False
}

_SMS_TRACKER = {
    "last_time": 0.0,
    "status": "IDLE", # IDLE, SENT, COOLDOWN
    "animal": ""
}

# ── Shared state ──────────────────────────────────────────────────────────────
_lock = Lock()
_video_camera: Any = None
_current_frame: Any = None
_alert_phone: Optional[str] = None

_detection: Dict[str, Any] = {
    "labels": [],
    "recent_alert": "",
    "confidence": 0.0,
    "pixel_area": 0,
    "box": [50, 50, 150, 150],
    "timestamp": 0.0,
}

_last_spoken: Dict[str, float] = {}
_sms_cooldowns: Dict[str, float] = {}
SPEAK_DELAY: float = 2.0
SMS_COOLDOWN: float = 60.0


# ── Camera ────────────────────────────────────────────────────────────────────
def _get_camera() -> Any:
    global _video_camera
    if _video_camera is None and _cv2 is not None:
        try:
            cam: Any = _cv2.VideoCapture(0)
            if cam.isOpened():
                cam.set(_cv2.CAP_PROP_FRAME_WIDTH, 640)
                cam.set(_cv2.CAP_PROP_FRAME_HEIGHT, 480)
                _video_camera = cam
            else:
                cam.release()
        except Exception as exc:
            print(f"[Camera] {exc}")
    return _video_camera


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


# ── SMS ───────────────────────────────────────────────────────────────────────
def _send_sms(phone: str, animal: str, pct: float) -> None:
    if twilio_client is None or TWILIO_PHONE is None:
        print(f"[SMS DEBUG] {animal} @ {pct:.1f}% → {phone}")
        return
    try:
        twilio_client.messages.create(
            body=f"ALERT: {animal} detected with {pct:.1f}% confidence in the zone.",
            from_=TWILIO_PHONE,
            to=phone,
        )
    except Exception as exc:
        print(f"[SMS] {exc}")


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
                    conf=_INF_CONFIG["conf"], 
                    iou=_INF_CONFIG["iou"], 
                    agnostic_nms=True,
                )
            except Exception as exc:
                print(f"[Inference] {exc}")
                time.sleep(0.01)
                continue

            animals: List[str] = []
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
                        # Filter by class depending on model type
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
                        animals.append(name)
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
                _detection["labels"] = list(set(animals))
                if found:
                    _detection["recent_alert"] = top_label
                    _detection["confidence"] = top_conf
                    _detection["pixel_area"] = top_area
                    _detection["box"] = top_box
                    _detection["timestamp"] = now

                    last_t: float = float(_last_spoken.get(top_label, 0.0))
                    if now - last_t > SPEAK_DELAY:
                        _last_spoken[top_label] = now
                        pct_int: int = int(top_conf * 100)
                        msg: str = f"{top_label.upper()} detected with {pct_int} percent accuracy"
                        # We could call _speak(msg) here if we wanted server-side voice.
                        # For now, we rely on the frontend voice to avoid duplication.

                    local_phone: Optional[str] = _alert_phone
                    if local_phone:
                        last_sms: float = float(_sms_cooldowns.get(top_label, 0.0))
                        if now - last_sms > SMS_COOLDOWN:
                            _sms_cooldowns[top_label] = now
                            _SMS_TRACKER["last_time"] = now
                            _SMS_TRACKER["status"] = "SENT"
                            _SMS_TRACKER["animal"] = top_label
                            Thread(target=_send_sms, args=(local_phone, top_label, top_conf * 100.0), daemon=True).start()
                else:
                    old_ts: float = float(_detection.get("timestamp", 0.0))
                    if now - old_ts > 2.0:
                        _detection["recent_alert"] = ""
                        _detection["confidence"] = 0.0
                        _detection["pixel_area"] = 0
                        _detection["box"] = [50, 50, 150, 150]
                        _detection["timestamp"] = 0.0

        time.sleep(0.01)


Thread(target=_inference_worker, daemon=True).start()


# ── Frame generator ───────────────────────────────────────────────────────────
def _generate_frames():
    global _current_frame, _video_camera
    while True:
        cam: Any = _get_camera()
        if cam is None or _cv2 is None:
            time.sleep(1)
            continue

        ok: bool
        frame: Any
        ok, frame = cam.read()  # type: ignore[misc]
        if not ok:
            try:
                cam.release()
            except Exception:
                pass
            _video_camera = None
            time.sleep(0.5)
            continue

        det: Dict[str, Any] = {}
        with _lock:
            _current_frame = frame.copy()
            det = _detection.copy()

        # ── HUD ───────────────────────────────────────────────────────────
        if _cv2 is not None:
            label_s: str = str(det.get("recent_alert", ""))
            if label_s:
                cf: float = float(det.get("confidence", 0.0))
                acc: float = int(cf * 1000) / 10.0
                raw_box: Any = det.get("box", [50, 50, 150, 150])
                if isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
                    try:
                        bx1 = int(float(raw_box[0]))
                        by1 = int(float(raw_box[1]))
                        bx2 = int(float(raw_box[2]))
                        by2 = int(float(raw_box[3]))
                        lbl: str = label_s.upper() if label_s else "DETECTING"
                        _cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
                        px: int = int(float(det.get("pixel_area", 0)))
                        _cv2.putText(frame, f"{lbl} {acc}% | {px}px",
                            (bx1, by1 - 10), _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    except Exception:
                        pass
            _cv2.putText(frame, "GALAXY SCAN  //  WEB STREAM",
                (10, 25), _cv2.FONT_HERSHEY_SIMPLEX, 0.55, (168, 85, 247), 2)

        enc: bool
        buf: Any
        if _cv2 is not None:
            try:
                enc, buf = _cv2.imencode(".jpg", frame, [int(_cv2.IMWRITE_JPEG_QUALITY), 70])
                if enc:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
                else:
                    time.sleep(0.1)
            except Exception:
                time.sleep(0.1)
        else:
            time.sleep(1)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def login_page() -> Any:
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register() -> Any:
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        
        if User.query.filter_by(email=email).first():
            flash("Email already registered")
            return redirect(url_for("register"))
            
        new_user = User(email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        
        login_user(new_user)
        return redirect(url_for("dashboard"))
        
    return render_template("register.html")

@app.route("/login", methods=["POST"])
def login() -> Any:
    email: Optional[str] = request.form.get("email")
    password: Optional[str] = request.form.get("password")
    
    user = User.query.filter_by(email=email).first()
    if user and user.check_password(password or ""):
        login_user(user)
        return redirect(url_for("dashboard"))
    
    flash("Invalid email or password")
    return redirect(url_for("login_page"))

@app.route("/logout")
@login_required
def logout() -> Any:
    logout_user()
    return redirect(url_for("login_page"))

@app.route("/dashboard")
@login_required
def dashboard() -> Any:
    return render_template("dashboard.html")

@app.route("/update_phone", methods=["POST"])
@login_required
def update_phone() -> Any:
    global _alert_phone
    body: Dict[str, Any] = request.get_json(silent=True) or {}
    raw_phone: Any = body.get("phone")
    if raw_phone:
        clean: str = str(raw_phone).strip().replace(" ", "")
        if not clean.startswith("+"):
            clean = "+" + clean
        with _lock:
            _alert_phone = clean
        return jsonify({"success": True, "message": "Phone registered"})
    return jsonify({"success": False, "message": "No phone provided"})

@app.route("/test_sms", methods=["POST"])
@login_required
def test_sms() -> Any:
    global _alert_phone
    if not _alert_phone:
        return jsonify({"success": False, "message": "No phone registered"})
    
    if not TWILIO_READY:
        # Mock success for user if credentials aren't real, but log it locally
        print(f"[SMS TEST] Mock sending to {_alert_phone} (Twilio not configured)")
        return jsonify({"success": True, "message": "Test initiated (Mock)"})
        
    try:
        twilio_client.messages.create(
            body="GALAXY SCAN: Alert protocol verified. Link stable.",
            from_=TWILIO_PHONE,
            to=_alert_phone,
        )
        return jsonify({"success": True, "message": "Protocol verified!"})
    except Exception as exc:
        print(f"[SMS Test Error] {exc}")
        return jsonify({"success": False, "message": f"Link Error: {str(exc)}"})

@app.route("/update_config", methods=["POST"])
@login_required
def update_config() -> Any:
    body: Dict[str, Any] = request.get_json(silent=True) or {}
    precise: bool = bool(body.get("precise", False))
    _INF_CONFIG["precise"] = precise
    if precise:
        _INF_CONFIG["conf"] = 0.45
        _INF_CONFIG["iou"] = 0.50
    else:
        _INF_CONFIG["conf"] = 0.30
        _INF_CONFIG["iou"] = 0.45
    return jsonify({"success": True, "precise": precise, "conf": _INF_CONFIG["conf"]})

@app.route("/video_feed")
@login_required
def video_feed() -> Any:
    return Response(_generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/latest_detection")
@login_required
def latest_detection() -> Any:
    with _lock:
        payload: Dict[str, Any] = _detection.copy()
        payload["twilio_ready"] = TWILIO_READY
        payload["phone_registered"] = bool(_alert_phone)
        payload["sms_tracker"] = _SMS_TRACKER.copy()
        payload["inf_config"] = _INF_CONFIG.copy()
    return jsonify(payload)

if __name__ == "__main__":
    app.run(debug=False, threaded=True, host="0.0.0.0", port=5000)
