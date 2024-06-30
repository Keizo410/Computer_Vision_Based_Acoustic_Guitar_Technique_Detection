import os
import cv2 as cv
import numpy as np
from ultralytics import YOLO


def segment_boxes_lr(model, input_folder, output_folder):
    expansion_factor = 0.01
    accumulated_mask_left = None  # Initialize accumulated mask for left hand
    accumulated_mask_right = None  # Initialize accumulated mask for right hand
    iou_threshold = 0.5

    # Ensure the output folder exists, create it if necessary
    os.makedirs(output_folder, exist_ok=True)

    # Define the classes you want to segment
    target_classes = ["left", "right"]

    # Iterate over MP4 files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".MP4"):
            input_video_path = os.path.join(input_folder, filename)
            output_video_path = os.path.join(output_folder, filename)

            accumulated_mask_left = None
            accumulated_mask_right = None

            # Open the input video file
            cap = cv.VideoCapture(input_video_path)

            # Check if the input video file was opened successfully
            if not cap.isOpened():
                print(f"Error: Could not open input video file {input_video_path}.")

            # Get the video properties
            frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            frame_rate = cap.get(cv.CAP_PROP_FPS)

            # Define the codec and create a VideoWriter object with the same frame rate
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            out = cv.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

            # Iterate over frames in the input video
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                result = model.predict(frame)
                mask_left = np.zeros(frame.shape[:2], dtype=np.uint8)
                mask_right = np.zeros(frame.shape[:2], dtype=np.uint8)

                for r in result:
                    img = np.copy(r.orig_img)

                    for target_class in target_classes:
                        target_detections = [d for d in r if d.names[d.boxes.cls.tolist().pop()] == target_class]

                        for c in target_detections:
                            box = c.boxes.xyxy[0]

                            expanded_box = expand_box(box, expansion_factor, frame_width, frame_height)
                            if target_class == "left":
                                cv.rectangle(mask_left, (int(expanded_box[0]), int(expanded_box[1])), (int(expanded_box[2]), int(expanded_box[3])), (255), cv.FILLED)
                            elif target_class == "right":
                                cv.rectangle(mask_right, (int(expanded_box[0]), int(expanded_box[1])), (int(expanded_box[2]), int(expanded_box[3])), (255), cv.FILLED)

                if accumulated_mask_left is None:
                    accumulated_mask_left = mask_left
                else:
                    resized_mask_left = cv.resize(mask_left, (accumulated_mask_left.shape[1], accumulated_mask_left.shape[0]))
                    accumulated_mask_left = cv.bitwise_or(accumulated_mask_left, resized_mask_left)

                if accumulated_mask_right is None:
                    accumulated_mask_right = mask_right
                else:
                    resized_mask_right = cv.resize(mask_right, (accumulated_mask_right.shape[1], accumulated_mask_right.shape[0]))
                    accumulated_mask_right = cv.bitwise_or(accumulated_mask_right, resized_mask_right)

                masked_frame_left = cv.bitwise_and(frame, frame, mask=accumulated_mask_left)
                masked_frame_right = cv.bitwise_and(frame, frame, mask=accumulated_mask_right)
                masked_frame_both = cv.bitwise_or(masked_frame_left, masked_frame_right)

                out.write(masked_frame_both)

            cap.release()
            out.release()

def segment_boxes_left(model, input_folder, output_folder):
    expansion_factor = 0.1
    accumulated_mask = None  # Initialize accumulated mask
    iou_threshold=0.5
    # Ensure the output folder exists, create it if necessary
    os.makedirs(output_folder, exist_ok=True)

    # Define the class you want to segment
    target_class = "left"

    # Iterate over MP4 files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):
            input_video_path = os.path.join(input_folder, filename)
            output_video_path = os.path.join(output_folder, filename)

            accumulated_mask = None

            # Open the input video file
            cap = cv.VideoCapture(input_video_path)

            # Check if the input video file was opened successfully
            if not cap.isOpened():
                print(f"Error: Could not open input video file {input_video_path}.")
                

            # Get the video properties
            frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            frame_rate = cap.get(cv.CAP_PROP_FPS)

            # Define the codec and create a VideoWriter object with the same frame rate
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            out = cv.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

            # Iterate over frames in the input video
            while True:
                # Read a frame from the input video
                ret, frame = cap.read()

                # Check if the frame was read successfully
                if not ret:
                    break

                # Run inference on the frame
                result = model.predict(frame)

                # Initialize an empty mask for the frame
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)

                # Iterate over detection results
                for r in result:
                    img = np.copy(r.orig_img)

                    # Filter detection results based on the target class
                    target_detections = [d for d in r if d.names[d.boxes.cls.tolist().pop()] == target_class]

                    # Iterate each object contour for the target class
                    for c in target_detections:
                        # Extract bounding box coordinates
                        box = c.boxes.xyxy[0]

                        expanded_box = expand_box(box, expansion_factor, frame_width, frame_height)

                        # Create a binary mask around the detected box
                        # cv.rectangle(mask, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255), cv.FILLED)
                        cv.rectangle(mask, (int(expanded_box[0]), int(expanded_box[1])), (int(expanded_box[2]), int(expanded_box[3])), (255), cv.FILLED)


                # Update accumulated mask
                if accumulated_mask is None:
                    accumulated_mask = mask
                else:
                    # Resize the current mask to match the dimensions of the accumulated mask
                    resized_mask = cv.resize(mask, (accumulated_mask.shape[1], accumulated_mask.shape[0]))
                    accumulated_mask = cv.bitwise_or(accumulated_mask, resized_mask)                               

                # Apply the accumulated mask to the frame
                masked_frame = cv.bitwise_and(frame, frame, mask=accumulated_mask)
                # Write the masked frame to the output video
                out.write(masked_frame)


            # Release the input and output video objects
            cap.release()
            out.release()

def segment_boxes_right(model, input_folder, output_folder):
    expansion_factor = 0.1
    accumulated_mask = None  # Initialize accumulated mask
    iou_threshold=0.5
    # Ensure the output folder exists, create it if necessary
    os.makedirs(output_folder, exist_ok=True)

    # Define the class you want to segment
    target_class = "right"

    # Iterate over MP4 files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):
            input_video_path = os.path.join(input_folder, filename)
            output_video_path = os.path.join(output_folder, filename)

            accumulated_mask = None

            # Open the input video file
            cap = cv.VideoCapture(input_video_path)

            # Check if the input video file was opened successfully
            if not cap.isOpened():
                print(f"Error: Could not open input video file {input_video_path}.")
                

            # Get the video properties
            frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            frame_rate = cap.get(cv.CAP_PROP_FPS)

            # Define the codec and create a VideoWriter object with the same frame rate
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            out = cv.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

            # Iterate over frames in the input video
            while True:
                # Read a frame from the input video
                ret, frame = cap.read()

                # Check if the frame was read successfully
                if not ret:
                    break

                # Run inference on the frame
                result = model.predict(frame)

                # Initialize an empty mask for the frame
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)

                # Iterate over detection results
                for r in result:
                    img = np.copy(r.orig_img)

                    # Filter detection results based on the target class
                    target_detections = [d for d in r if d.names[d.boxes.cls.tolist().pop()] == target_class]

                    # Iterate each object contour for the target class
                    for c in target_detections:
                        # Extract bounding box coordinates
                        box = c.boxes.xyxy[0]

                        expanded_box = expand_box(box, expansion_factor, frame_width, frame_height)

                        # Create a binary mask around the detected box
                        # cv.rectangle(mask, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255), cv.FILLED)
                        cv.rectangle(mask, (int(expanded_box[0]), int(expanded_box[1])), (int(expanded_box[2]), int(expanded_box[3])), (255), cv.FILLED)
                                
                # Apply the accumulated mask to the frame
                masked_frame = cv.bitwise_and(frame, frame, mask=accumulated_mask)
                # Write the masked frame to the output video
                out.write(masked_frame)


            # Release the input and output video objects
            cap.release()
            out.release()

def get_right(model, input_folder, output_folder):
    # Ensure the output folder exists, create it if necessary
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over MP4 files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):
            input_video_path = os.path.join(input_folder, filename)
            output_video_path = os.path.join(output_folder, filename)

            # Open the input video file
            cap = cv.VideoCapture(input_video_path)

            # Check if the input video file was opened successfully
            if not cap.isOpened():
                print(f"Error: Could not open input video file {input_video_path}.")
                continue

            # Get the video properties
            frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            frame_rate = cap.get(cv.CAP_PROP_FPS)

            # Define the codec and create a VideoWriter object with the same frame rate
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            out = cv.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

            # Iterate over frames in the input video
            while True:
                # Read a frame from the input video
                ret, frame = cap.read()

                # Check if the frame was read successfully
                if not ret:
                    break

                # Initialize a mask with all ones (no masking)
                mask = np.ones_like(frame[:, :, 0], dtype=np.uint8)

                # Mask out the right side of the frame
                mask[:, frame_width // 2:] = 0

                # Apply the mask to the frame
                masked_frame = cv.bitwise_and(frame, frame, mask=mask)

                # Write the masked frame to the output video
                out.write(masked_frame)

            # Release the input and output video objects
            cap.release()
            out.release()

def expand_box(box, expansion_factor, frame_width, frame_height):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    new_x1 = max(0, int(x1 - expansion_factor * width))
    new_y1 = max(0, int(y1 - expansion_factor * height))
    new_x2 = min(frame_width, int(x2 + expansion_factor * width))
    new_y2 = min(frame_height, int(y2 + expansion_factor * height))
    return new_x1, new_y1, new_x2, new_y2

def main():

    input_folder = "../../../guitar_technique_detection/data/actions_dataset/data/original"
    output_folder= "../../../guitar_technique_detection/data/actions_dataset/data/segmented"

    # Load a model
    model = YOLO('../../utils/yolo_hand_detection_training/runs/detect/train6/weights/best.pt')

    for folder in os.listdir(input_folder):
        input = os.path.join(input_folder, folder)
        output = os.path.join(output_folder, folder)
        segment_boxes_left(model, input, output)

if __name__ == "__main__":
    main()