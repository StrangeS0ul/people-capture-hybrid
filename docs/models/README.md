# Models for the browser demo
Place a YOLOv8 detection ONNX file here named `yolov8n.onnx`.

Export example:
```python
from ultralytics import YOLO
YOLO('yolov8n.pt').export(format='onnx', opset=12, dynamic=False)
```
