import os
import yaml
from pathlib import Path
import torch
from ultralytics import YOLO


torch.cuda.empty_cache()

def main():
    BASE_DIR = Path.cwd().parent.parent  

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    with open(BASE_DIR / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    data = BASE_DIR / config["data_path"]

    def train(dataset, epochs):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Device used: ", device)

        save_dir = BASE_DIR / config["yolo_finetuned_weight_output_path"]

        model = YOLO("yolov8n.pt").to(device) 
        model.train(data=dataset, epochs=epochs, workers=0, save_dir=save_dir)
        print("Model is using device:", model.device)

    epochs = 200
    train(data, epochs)

if __name__ == "__main__":
    main()
