# DefenceVision AI: Object Detection Using Deep Learning

## Abstract
DefenceVision AI is an educational computer-vision application that detects common objects in user-uploaded images. It uses a pretrained YOLOv8 nano model and a Streamlit interface to display bounding boxes, class labels, confidence scores and object analytics.

## Problem Statement
Identifying and counting objects manually can take time. Object-detection models can identify supported classes and estimate their positions in an image. This project demonstrates that workflow in an interactive web application.

## Objectives
- Build an image-upload interface.
- Apply a pretrained object-detection model.
- Visualize object locations and labels.
- Display confidence scores and counts.
- Allow users to download annotated images.

## Technologies
Python, Streamlit, Ultralytics YOLOv8 nano, Pillow and Pandas.

## Methodology
1. Accept an image from the user.
2. Load the pretrained YOLO model.
3. Run inference with a selected confidence threshold.
4. Extract class, confidence and bounding-box coordinates.
5. Draw boxes and labels over the image.
6. Display image, metrics and a table.
7. Allow annotated output to be downloaded.

## Limitations
The model can miss objects or produce false detections. Results depend on image quality and the classes represented in its training data. Confidence is not a guarantee of correctness. This is not a validated operational defence or safety-critical system.

## Future Scope
Evaluate on a labelled dataset, compare model variants, add batch processing, export summaries, and calculate precision, recall and mean average precision.

## Conclusion
DefenceVision AI demonstrates an end-to-end computer-vision workflow in a user-friendly Python web application.

## Viva Questions
**1. What is object detection?** Identifying object classes and their locations in an image.

**2. What is YOLO?** You Only Look Once, a family of object-detection models.

**3. What is a bounding box?** A rectangle showing the estimated location of an object.

**4. What is a confidence score?** A model score associated with a predicted class; it is not a guarantee.

**5. Why use a pretrained model?** It avoids training a large model from scratch.

**6. What does Pillow do?** Loads and draws annotations on images.

**7. How can performance be evaluated?** On labelled data using precision, recall, mAP and inference time.

**8. What are the limitations?** Missed objects and false detections, especially for poor-quality or unfamiliar images.
