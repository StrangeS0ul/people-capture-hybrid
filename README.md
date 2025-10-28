# People Capture Hybrid — MediaPipe + YOLO

A rebuildable project that fuses **MediaPipe Pose + Hands** with optional **YOLOv8 segmentation** on the desktop, and provides a **browser demo** (ONNX Runtime Web + MediaPipe Pose) for quick try‑outs.

- **Desktop (Python/OpenCV):** multi‑person pose + hands, wrist→hand assignment, convex hull, EMA smoothing, majority coverage, optional YOLOv8‑seg or SelfieSeg for outline.
- **Browser (GitHub Pages):** YOLOv8 ONNX detection + MediaPipe Pose via CDN, with a simple fusion rule to confirm boxes using landmarks.

## Why combine both
Boxes are fast and robust to clutter. Landmarks encode human structure. Confirming detections with landmarks reduces false positives and produces steadier tracks. This hybrid approach is explainable and easy to tune.

---

## Quickstart — Desktop (recommended first)
1. Create and activate a Python 3.10+ environment.

2. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
3. Run:
   ```bash
   python app.py --camera 0 --width 1280 --height 720
   ```

Notes:
- The script will auto‑download the required MediaPipe `.task` models to `./models/` on first run.
- YOLOv8‑seg is optional. If Ultralytics isn’t installed or `--no-seg` is used, it falls back to SelfieSeg.

### Optional: enable YOLOv8 segmentation
```bash
pip install ultralytics
# Uses yolov8n-seg by default (in code). Change model path if needed.
python app.py
```

---

## Browser demo — GitHub Pages
The `/docs` folder hosts a client‑side demo. It runs **ONNX Runtime Web** and **MediaPipe Pose** in the browser.

### Steps
1. Export a YOLOv8 detection model to ONNX (Ultralytics):

   ```python
   from ultralytics import YOLO
   YOLO('yolov8n.pt').export(format='onnx', opset=12, dynamic=False)
   ```
2. Place the ONNX at `docs/models/yolov8n.onnx`.
3. Serve locally:
   ```bash
   python -m http.server -d docs 8000
   ```
4. Open `http://localhost:8000` and allow the camera.
5. To publish: push to GitHub and enable **Pages → Deploy from branch → main /docs**.

---

## Fusion logic (browser)
- YOLOv8 gives person boxes + scores.
- MediaPipe Pose gives landmarks.
- Confirmed if ≥8 visible landmarks fall inside the box.
- Fused score = 0.7×yolo + 0.3×pose_visibility; unconfirmed uses 0.5×yolo.
- NMS on fused scores.

You can tweak thresholds in `docs/app.js`.

---

## Repo layout
```
people-capture-hybrid/
├─ app.py                  # Desktop capture: multi-person pose+hands + optional YOLOv8-seg
├─ requirements.txt
├─ LICENSE
├─ README.md
├─ models/                 # Auto-downloaded MP .task files on first run
├─ scripts/
│  └─ export_yolov8_to_onnx.py
└─ docs/                   # Browser demo (GitHub Pages)
   ├─ index.html
   ├─ app.js
   ├─ style.css
   └─ models/
      └─ README.md
```

---

## Benchmarks (fill after testing)
Add a table like:
| Mode | HW | FPS | Latency (ms) | Notes |
|---|---|---:|---:|---|
| Desktop (pose+hands) | i7 + iGPU | 30 | 33 | SelfieSeg |
| Desktop (pose+hands+YOLO-seg) | i7 + RTX 3060 | 28 | 36 | yolov8n-seg |
| Browser (YOLOv8n + Pose) | Laptop Chrome | 12 | 83 | WebGL |

---

## License
MIT
