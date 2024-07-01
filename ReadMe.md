# Acoustic Guitar Technique Detection Model 

## Prerequisite

- tbd

## Implementation

### 1. Data Processing & Feature Engineering

#### 1.1 Hand Detection
Train a YOLO model for detecting hands in video frames.

#### 1.2 Hand Segmentation
Use the trained YOLO model to segment hands from video frames.

#### 1.3 Optical Flow Extraction
Extract optical flow from the segmented video clips and store the normalized motion vectors as histograms of motion for corresponding directions.

### 2. Shallow Neural Networks Classification

Train a shallow neural network classifier (Multi LeNet) to classify the guitar techniques based on the histograms of motion vectors.

## How to Use

### Step 1
Train YOLO for hand detection and segmentation using the script `yolo_train.py`.

```bash
python yolo_train.py
```

### Step 2
Segment hands from video clips for each guitar technique using the trained YOLO model.

```python
# Your segmentation script here, for example:
python segment_hands.py 
```

### Step 3
Extract optical flow from each video clip and store the normalized motion vectors as histograms of motion for the corresponding direction.

```python
# Your optical flow extraction script here, for example:
python optical_flow_extraction.py 
```

### Step 4
Train the Multi LeNet classifier to classify the guitar techniques based on the histograms.

```python
# Your training script here, for example:
python train_mul.py 
```

## Summary

This project involves detecting and segmenting hands from video clips, extracting optical flow to create histograms of motion, and training a neural network classifier to detect different acoustic guitar techniques. Follow the steps outlined above to preprocess your data, extract features, and train your classifier.
