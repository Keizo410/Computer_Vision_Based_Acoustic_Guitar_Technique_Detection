from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
from ultralytics import YOLO
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def train(dataset, epochs):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device used: ", device)

    model = YOLO("yolov8n.pt").to(device) 
    model.train(data=dataset, epochs=epochs)  
    metrics = model.val()  
    print("Model is using device:", model.device)

def main():
  # Should be full path, otherwise errors
  data = "C:\\Users/iek42\\Projects\\guitar_technique_detection\\data\\hand_image_dataset\\data.yaml"  
  epochs = 200
  train(data, epochs)

if __name__ == '__main__':
    main()
