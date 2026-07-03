# AI-Based Traffic Signal Violation Detection System

A deep learning-based computer vision system for detecting potential traffic signal violations from fixed traffic cameras.

The system combines a **multi-task Convolutional Neural Network (CNN)** with **rule-based decision logic** to identify red-light violations. Instead of directly predicting whether a vehicle has violated the signal, the model first learns to:

- Detect vehicle locations
- Classify the traffic signal state

A rule-based module then determines whether a violation has occurred by checking if a detected vehicle overlaps a predefined stop-line or zebra-crossing region while the traffic signal is red.

---

# Features

- Multi-task CNN architecture
- Traffic signal classification
  - Red
  - Yellow
  - Green
  - Unknown
- Vehicle localization using bounding box regression
- Rule-based violation detection
- Custom PyTorch Dataset
- Multi-task loss function
- Automatic model checkpointing
- Training loss visualization
- Signal confusion matrix generation
- GPU (CUDA) support

---

# System Architecture

```
                    Input Image
                         │
                         ▼
              CNN Feature Extractor
                         │
                         ▼
             Shared Feature Representation
                    (1024 Features)
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
 Traffic Signal Head             Vehicle Detection Head
(Classification)                (Bounding Box Regression)
         │                               │
         ▼                               ▼
Signal State                  Vehicle Bounding Boxes
         │                               │
         └───────────────┬───────────────┘
                         ▼
               Rule-Based Decision Logic
                         │
                         ▼
          Red Light Violation Detection
```

---

# How It Works

The system operates in three stages.

## Stage 1 — Traffic Signal Classification

The CNN predicts the traffic signal state as:

- Red
- Yellow
- Green
- Unknown

---

## Stage 2 — Vehicle Detection

The same CNN predicts the locations of vehicles in the image.

Each predicted bounding box contains:

```
[x_center, y_center, width, height, confidence]
```

---

## Stage 3 — Rule-Based Violation Detection

The CNN **does not directly predict violations**.

Instead, a deterministic rule is applied.

```
IF

Traffic Signal == RED

AND

Vehicle Bounding Box overlaps Stop Line / Zebra Crossing

THEN

Violation = TRUE
```

This makes the system explainable and easier to validate compared to an end-to-end violation classifier.

---

# Dataset Format

The dataset should be provided as a CSV file.

Required columns:

| Column | Description |
|---------|-------------|
| image_path | Image location |
| x | Bounding box X coordinate |
| y | Bounding box Y coordinate |
| w | Bounding box width |
| h | Bounding box height |
| class_id | Vehicle class ID |
| class_name | Vehicle class name |
| signal_state | Traffic signal state |

Example:

```csv
image_path,x,y,w,h,class_id,class_name,signal_state
frame001.jpg,120,230,80,60,1,vehicle,red
frame001.jpg,350,210,90,70,1,vehicle,red
```

---

# Model Architecture

The model consists of three components.

## 1. Shared CNN Backbone

The CNN extracts visual features from the input image.

Architecture:

```
Conv → BatchNorm → ReLU → MaxPool

3 → 32 channels

↓

32 → 64

↓

64 → 128

↓

128 → 256

↓

256 → 512

↓

Adaptive Average Pooling

↓

Shared Fully Connected Layer
```

---

## 2. Signal Classification Head

Predicts:

- Red
- Yellow
- Green
- Unknown

Uses:

- Fully Connected Layer
- Cross Entropy Loss

---

## 3. Vehicle Detection Head

Predicts up to **10 vehicles**.

Each prediction contains:

```
x_center
y_center
width
height
confidence
```

Uses:

- Fully Connected Layers
- Sigmoid Activation

---

# Loss Function

A custom multi-task loss combines three objectives.

## Bounding Box Loss

Smooth L1 Loss

Purpose:

- Improve vehicle localization accuracy

---

## Confidence Loss

Binary Cross Entropy Loss

Purpose:

- Predict whether each box contains a vehicle

---

## Signal Classification Loss

Cross Entropy Loss

Purpose:

- Predict traffic signal color

---

## Total Loss

```
Total Loss =
5 × Bounding Box Loss
+ Confidence Loss
+ Signal Classification Loss
```

---

# Training Pipeline

During training:

1. Load images
2. Resize images
3. Normalize pixel values
4. Generate bounding box targets
5. Generate signal labels
6. Forward pass
7. Compute custom multi-task loss
8. Backpropagation
9. Update weights using Adam optimizer
10. Validate model
11. Save checkpoints

---

# Evaluation

The model is evaluated using:

- Validation Loss
- Bounding Box Loss
- Confidence Loss
- Signal Classification Loss
- Traffic Signal Confusion Matrix

Training also generates:

- Loss curves
- Confusion matrix
- Training log

---

# Project Structure

```
project/

│
├── annotations.csv
├── raw_frames/
│
├── train_fixed_camera_rule_cnn.py
│
├── saved_models_rule/
│   ├── best_model.pth
│   ├── latest_model.pth
│   ├── training_log.csv
│   └── graphs/
│       ├── loss_curves.png
│       └── signal_confusion_matrix.png
│
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/traffic-violation-detection.git
```

Install dependencies

```bash
pip install torch torchvision
pip install opencv-python
pip install numpy
pip install pandas
pip install matplotlib
```

---

# Training

Run:

```bash
python train_fixed_camera_rule_cnn.py \
    --csv annotations.csv \
    --epochs 30 \
    --batch-size 8 \
    --max-boxes 10
```

---

# Outputs

Training generates:

```
saved_models_rule/

best_model.pth

latest_model.pth

training_log.csv

graphs/

loss_curves.png

signal_confusion_matrix.png
```

---

# Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- Pandas
- Matplotlib
- CUDA
- Deep Learning
- Computer Vision
- Convolutional Neural Networks (CNN)
- Multi-Task Learning

---

# Future Improvements

- Replace fixed bounding-box prediction with YOLOv8 or Faster R-CNN
- Detect traffic lights using object detection instead of image-level classification
- Add vehicle tracking using DeepSORT or ByteTrack
- Support multiple traffic lanes
- Real-time video inference
- Automatic stop-line detection
- Improve robustness under adverse weather and nighttime conditions
- Deploy as a REST API or edge application

---

# Applications

- Intelligent Traffic Monitoring
- Smart Cities
- Automated Traffic Law Enforcement
- Traffic Analytics
- Urban Transportation Systems
- Road Safety Monitoring

---

# Author

**Ali Naeem**

Computer Vision • Artificial Intelligence • Machine Learning • Deep Learning

```

This README is suitable for a GitHub portfolio and clearly explains the problem, architecture, training process, and technical decisions in a way that recruiters, professors, and other developers can quickly understand.
