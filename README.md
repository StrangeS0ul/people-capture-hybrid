# People Capture Hybrid — MediaPipe + YOLO

A rebuildable project that fuses **MediaPipe Pose + Hands** with optional **YOLOv8 segmentation** on the desktop, and provides a **browser demo** (ONNX Runtime Web + MediaPipe Pose) for quick try‑outs.

- **Desktop (Python/OpenCV):** multi‑person pose + hands, wrist→hand assignment, convex hull, EMA smoothing, **temporal hysteresis**, majority coverage, optional YOLOv8‑seg or SelfieSeg for outline.
- **Browser (GitHub Pages):** YOLOv8 ONNX detection + MediaPipe Pose via CDN with **client‑side fusion**, **frame‑skipping**, and **on/off toggles**.

Live demo: https://stranges0ul.github.io/people-capture-hybrid/

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
- Auto‑detects GPU. Force with `--device cuda:0` or `--device cpu`.
- Toggle modules: `--no-hands`, `--no-yolo`, `--flip`.
- MediaPipe `.task` models auto‑download to `./models/` on first run.

### Optional: enable YOLOv8 segmentation
```bash
pip install ultralytics
# Uses yolov8n-seg by default (in code). Change model path if needed.
python app.py
```

---

## Browser demo — GitHub Pages
The `/docs` folder hosts a client‑side demo. It runs **ONNX Runtime Web** and **MediaPipe Pose** in your browser (camera stays local).

### Steps
1. Export YOLOv8 detection to ONNX (Ultralytics):
   ```python
   from ultralytics import YOLO
   YOLO('yolov8n.pt').export(format='onnx', opset=12, dynamic=False)
   ```
2. Place the model at `docs/models/yolov8n.onnx`.
3. Serve locally:
   ```bash
   python -m http.server -d docs 8000
   ```
4. Open `http://localhost:8000` and allow the camera.
5. Publish on GitHub Pages:
   - Repo Settings → Pages → **Deploy from a branch** → Branch: `main` → Folder: `/docs`.
   - Ensure `docs/.nojekyll` exists and `docs/models/yolov8n.onnx` is **committed** (not ignored).

---

## Browser controls and toggles
Right‑side panel in the demo.

- **Use YOLO** — turn detector on/off.  
  If you turn this **off**, also turn **Draw boxes** off.
- **Use Pose** — turn landmarks on/off.  
  If you turn this **off**, also turn **Draw pose** off.
- **YOLO every N** — run detector every N frames to reduce stutter.
- **Pose every N** — run pose every N frames to reduce load.
- **TopK** — keep top‑K candidates before NMS (lower = faster).
- **YOLO conf** — detection confidence threshold.
- **NMS IoU** — merge overlapping boxes; higher = fewer.
- **Draw boxes** / **Draw pose** — overlay rendering toggles.

**Recommended defaults (laptop):** YOLO every **2**, Pose every **2**, TopK **50**, YOLO conf **0.55**, NMS IoU **0.45**.  
If FPS drops, raise **YOLO every** to 3–4 and lower **TopK**.

---

## Fusion logic (browser)
- YOLOv8 → person boxes + scores.  
- MediaPipe Pose → 33 landmarks + visibility.  
- “Confirmed” if ≥ **8** visible landmarks fall inside a box.  
- Fused score = `0.7*yolo + 0.3*pose_visibility`; unconfirmed uses `0.5*yolo`.  
- NMS runs on **de‑letterboxed** video‑space coordinates.

Thresholds live in `docs/app.js`.

---

## Repo layout
```
people-capture-hybrid/
├─ app.py                  # desktop: pose+hands (+YOLO seg optional), tracker+hysteresis
├─ requirements.txt
├─ LICENSE
├─ README.md
├─ models/                 # MediaPipe .task files (auto-downloaded)
├─ scripts/
│  └─ export_yolov8_to_onnx.py
└─ docs/                   # browser demo (GitHub Pages)
   ├─ index.html
   ├─ app.js              # toggles, frame-skipping, fusion
   ├─ style.css
   └─ models/
      └─ yolov8n.onnx
```

---

## Benchmarks
| Mode | HW | FPS | Latency (ms) | Notes |
|---|---|---:|---:|---|
| Desktop (pose+hands) | i7 + iGPU | 32 | 34 | SelfieSeg |
| Desktop (pose+hands+YOLO-seg) | i7 + RTX 3060 | 45 | 24 | yolov8n-seg |
| Browser (YOLOv8n + Pose) | Laptop Chrome | 16-18 | 60 | WebGL |

---

## Troubleshooting on Local Desktop
- **No camera prompt:** try `http://127.0.0.1:8000` and allow Camera in site settings.  
- **ONNX 404 on Pages:** commit `docs/models/yolov8n.onnx`, add `docs/.nojekyll`, and push.  
- **Slow FPS:** increase **YOLO every** and **Pose every**; lower **TopK**.  
- **Git LFS warning:** don’t store the ONNX in LFS; Pages won’t serve LFS objects.

---

## License
MIT
