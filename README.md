# DefenceVision AI — Object Detection Mini Project

## Features
- Dark defence-tech dashboard with navigation
- Upload JPG, PNG, JPEG or WEBP images
- Detect common objects using pretrained YOLOv8 nano
- Bounding boxes, labels, confidence scores and object counts
- Analytics page and annotated-image download
- Project details and viva notes

## Windows PowerShell setup
1. Extract the ZIP and open the `DefenceVision_AI` folder in VS Code.
2. Select **Terminal → New Terminal**.
3. Run these commands one at a time:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If `py` is not recognized, use `python -m venv .venv`. Open the local URL shown, usually `http://localhost:8501`.

## First-run note
The first run may download `yolov8n.pt` model weights, so internet access is required. Installing Ultralytics can take several minutes.

## How it works
The user uploads an image; YOLO predicts class labels, bounding boxes and confidence scores; Pillow draws the annotations; Streamlit displays the results and offers a PNG download.

## Responsible use
Educational prototype for user-uploaded images only. It does not perform face recognition, identity tracking, live surveillance or operational targeting. Predictions can be wrong and should not be used for safety-critical decisions.
