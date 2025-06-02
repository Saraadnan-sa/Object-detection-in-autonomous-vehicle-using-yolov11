# Object Detection with YOLOv11 - Self-Driving Cars

## Project Overview

This Jupyter Notebook facilitates object detection tasks for self-driving cars using the **YOLOv11** model. It covers environment setup, dataset handling from local Kaggle uploads, and preparing for model training and evaluation.

## Contents

* **Environment Setup**: Verifies GPU availability and installs required packages.
* **Dataset Preparation**: Uses locally uploaded dataset files (images and labels) for object detection training.
* **Directory Configuration**: Organizes input/output folders for training, validation, and testing.

## Prerequisites

* **Python 3.x**
* **Jupyter Notebook** (executed in Kaggle or similar environment)
* **YOLOv11** (via `ultralytics`)
* **CUDA** (automatically used in Kaggle GPU notebooks)

## How to Run

1. **Upload Dataset**:

   * Download the dataset from [Roboflow Self-Driving Car Dataset](https://public.roboflow.com/object-detection/self-driving-car).
   * Upload it to your Kaggle notebook environment (usually appears under `/kaggle/input/<dataset-folder>`).

2. **Launch the Notebook**:
   Open and run `Self_driving_vehicles.ipynb` in Kaggle.

3. **Execute the Notebook Cells**:
   Run each cell sequentially to configure the environment, load the dataset, train the model, and visualize results.

## Dataset

* **Source**: Roboflow (manually downloaded)
* **Structure** (after unzipping/uploading):

  ```
  /kaggle/input/self-driving-car/
      ├── train/
      │   ├── images/
      │   └── labels/
      ├── valid/
      │   ├── images/
      │   └── labels/
      └── data.yaml
  ```
* Ensure that the `data.yaml` file includes paths relative to the Kaggle environment.

## Outputs

* **Trained Model**: Saved to `/kaggle/working/yolov11/runs/detect/`.
* **Logs & Metrics**: Training progress, mAP, precision, recall, etc., viewable in real time in the notebook.

## Architecture Diagram

![image](https://github.com/user-attachments/assets/2ed4b403-97e0-4747-ab4d-e561db6e9889)


### Components:

* **Backbone**: Extracts key image features.
* **Neck**: Fuses features at different scales.
* **Head**: Outputs bounding boxes and class scores.

## Testing

After training completes, test YOLOv11 on sample images:

```python
from ultralytics import YOLO

model = YOLO("/kaggle/working/yolov11/runs/detect/train/weights/best.pt")
results = model("/kaggle/input/self-driving-car/test/images/sample.jpg", show=True)
```

Example Output:
![val_batch2_labels](https://github.com/user-attachments/assets/6c3caf38-4e01-473c-8980-2a5b1d35431e)

## Results

Model evaluation will generate:

* mAP (mean Average Precision)
* Precision/Recall
* Confusion Matrix
* Sample Detections

Results:
![results](https://github.com/user-attachments/assets/b330c303-c723-4b54-800e-e8414ee66210)

## Acknowledgements

* **YOLOv11** by Ultralytics
* **Roboflow**: For dataset provisioning
* **Kaggle**: For GPU resources and hosting environment
