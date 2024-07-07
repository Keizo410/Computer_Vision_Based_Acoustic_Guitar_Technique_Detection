# Acoustic Guitar Technique Detection Model 
## About
This repository contains a computer-vision-based approach for detecting acoustic guitar techniques (Hammer-on, Pull-off, and Slide). This project has two sections for this project: __Data Preparation__ and __Model Training__. 
### Subdirectories: 
  * data - contains hand image, hand action, histogram of motion datasets.
  * model - contains the checkpoint for multi-stream CNN.
  * utils - contains each module to process data and re-examine or integrate for the future edition. 

## Prerequisite

- tbd

- yolo train.py need full path for yaml file as well as yaml file's data path should be adjusted.

## Implementation

### 1. Data Processing & Feature Engineering

#### 1.1 Hand Detection
Train a YOLO model for detecting hands in video frames.

#### 1.2 Hand Segmentation
Use the trained YOLO model to segment hands from video frames.
- Note: To deal with the case when the model didn't detect any hand in a frame, the hand segmentation box is set to be accumulative, meaning once the box is drawn on the screen it'll be carried to the next frame. In this way, we can prevent having the black frame due to the hand detection model's miss detection, which causes some trouble for the next optical flow extraction step. This could be removed once the hand detection model has more data to retrain for higher precision & recall. 

#### 1.3 Optical Flow Extraction
Extract optical flow from the segmented video clips and store the normalized motion vectors as histograms of motion for corresponding directions.
- Note: Histograms of motion contain relative frequency over distance for corresponding direction, which are left, up left, up, up right, right, down right, down, down left, and no direction (not enough distance to determine the direction). Also, to get the main features of each technique motion, I set a minimum distance to be stored for histogram generation. 
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

## Note
Although box-based feature points for optical flow extraction contributed to extracting overall hand motion in the detected box, other methods such as media pipe's hand landmark detection directly contribute to better hand motion extraction. Due to some compatibility issue, it is not implemented in this project; however, I probably add it once this project is dockerized or found compatibility with it.
