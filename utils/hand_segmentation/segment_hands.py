import os
import cv2 as cv
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import yaml 

def expand_box(box_shape, expansion_factor, frame_width, frame_height):
    """
    A method to expand input box by multipling by expansion_factor.

    args:
    box_shape(int,int,int,int): box shape left top (x, y) and right bottom (x, y).
    expansion_factor(int): expansion factor to expand segmented box.
    frame_width(int): orignal frame widht
    frame_height(int): original frame height
    """
    x1, y1, x2, y2 = box_shape
    width = x2 - x1
    height = y2 - y1
    new_x1 = max(0, int(x1 - expansion_factor * width))
    new_y1 = max(0, int(y1 - expansion_factor * height))
    new_x2 = min(frame_width, int(x2 + expansion_factor * width))
    new_y2 = min(frame_height, int(y2 + expansion_factor * height))
    return new_x1, new_y1, new_x2, new_y2
    
def get_target_detection_result(results, target_class):
    """
    A method to extract target class from detection results.

    args:
    results
    target_class(str): target class for detection.
    """
    target_detections = []
    for detection in results:
        class_indices = detection.boxes.cls.tolist()
        if class_indices:  
            class_index = int(class_indices[0])  
            class_name = detection.names[class_index]  # Get class name
    
            if class_name == target_class:
                target_detections.append(detection)
    return target_detections

    
def segment_hand(model, input_folder, output_folder, expansion_factor = 0.1, target_class = "left"):
    """
    A method to segment an object from background by going through each video frame. 
    To minimize the effect of hand detection failure, segmentation box expansion is implemented,
    which keeps the detected hand box even the detection model missed and "grow" the box accumulatively 
    when the object moved out of the previous box area. 

    args: 
    target_class(str): a target class to detect and segment.
    model(str): hand detection model weight path.
    input_folder(str): input video folder path
    output_folder(str):  output video folder path
    expansion_factor(float): a float number for box expansion factor.
    """
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):
            input_video_path = os.path.join(input_folder, filename)
            output_video_path = os.path.join(output_folder, filename)
            
            cumulative_mask = None

            cap = cv.VideoCapture(input_video_path)

            if not cap.isOpened():
                print(f"Error: Could not open input video file {input_video_path}.")

            frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            frame_rate = cap.get(cv.CAP_PROP_FPS)

            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            out = cv.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                results = model.predict(frame)
                
                #frame.shape: [width, height, channel]
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)

                for result in results:
                    target_detections = get_target_detection_result(result, target_class)
                    for contour in target_detections:
                        box = contour.boxes.xyxy[0] # bounding box coordinates
                        expanded_box = expand_box(box, expansion_factor, frame_width, frame_height)
                        cv.rectangle(mask, (int(expanded_box[0]),int(expanded_box[1])),(int(expanded_box[2]),int(expanded_box[3])), (255), cv.FILLED)

                if cumulative_mask is None:
                   cumulative_mask = mask
                else:
                    CUMULATIVE_MASK_HEIGHT = cumulative_mask.shape[0]
                    CUMULATIVE_MASK_WIDTH = cumulative_mask.shape[1]
                    resized_mask = cv.resize(mask, (CUMULATIVE_MASK_WIDTH, CUMULATIVE_MASK_HEIGHT))
                    cumulative_mask = cv.bitwise_or(cumulative_mask, resized_mask)

                masked_frame = cv.bitwise_and(frame, frame, mask = cumulative_mask)
                out.write(masked_frame)

            cap.release()
            out.release()

def main():

    BASE_DIR = Path.cwd().parent.parent
    
    with open(BASE_DIR / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    weights_path = BASE_DIR / config["detection_weight_path"]
    input_folder = BASE_DIR / config["action_input_data_path"]
    output_folder= BASE_DIR / config["segmentation_output_data_path"]

    model = YOLO(weights_path)
    for folder in os.listdir(input_folder):
        input = os.path.join(input_folder, folder)
        output = os.path.join(output_folder, folder)
        segment_hand(model, input, output)

if __name__ == "__main__":
    main()