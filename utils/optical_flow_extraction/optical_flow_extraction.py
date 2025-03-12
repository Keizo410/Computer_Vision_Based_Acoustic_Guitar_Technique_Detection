import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2 as cv
from pathlib import Path
import yaml 
import numpy as np 
import matplotlib.pyplot as plt

BASE_DIR = Path.cwd().parent.parent

with open(BASE_DIR / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

histogram_output_folder_path = BASE_DIR / config["histogram_of_motion_output_data_path"]

MAX_FRET_MOVEMENT_THRESHOLD = 18 
MIN_MOVEMENT_DISTANCE = 0.01

def check_fret_movement(down_direction, up_direction):
    """
    A method to detect the stored movement is due to fret movement, not finger movement.

    args:
    down_direction(int): 
    up_direction(int): 
    return:
    boolean: True if fretmovement is detected, otherwise False. 
    """
    if down_direction < MAX_FRET_MOVEMENT_THRESHOLD or up_direction < MAX_FRET_MOVEMENT_THRESHOLD:
        return False
    return True
    
def eliminate_displacement(displacement, threshold):
    """
    A method to eliminate a small displacement by selected threshold.

    args: 
    displacement(int): a size of displacement vector
    threshold(int): a int value for thresholding

    return:
    displacement(int): processed displacement
    """
    if displacement < threshold:
        return 0
    return displacement
    
def calculate_direction(old_points, new_points):
    """
    Calculate the direction from an old point to a new point.

    param
    old_point: Coordinates of the old point as a tuple (x, y).
    new_point: Coordinates of the new point as a tuple (x, y).
    
    return: Angle (in radians) representing the direction from the old point to the new point.
    """
    displacement = np.array(new_points) - np.array(old_points)
    distance = np.linalg.norm(displacement)

    if distance < MIN_MOVEMENT_DISTANCE:
        return distance, 8

    direction_rad = np.arctan2(displacement[1], displacement[0])
    direction_deg = np.degrees(direction_rad)

    if direction_deg < 0:
        direction_deg += 360

    direction_section = int(direction_deg/45)

    return distance, direction_section

def create_histgram_of_motion(dictionary, class_name, case_num = None, frame_count = None):
    """
    Create histogram for each direction. x axis is magnitude and y axis is frequencies.
    0 : Right
    1 : Up Right
    2 : Up
    3 : Up Left
    4 : Left
    5 : Down Left
    6 : Down
    7 : Down Right
    8 : No Direction

    args:
    dictionary(python dict): a python dictionary containing distances for each direction.
    class_name(str): a class of the action type
    case_num(int): a case number of the video  
    frame_count(int): a total number of frames within a video
    """
    for key, distances in dictionary.items():

        if len(distances) > 0:
            max_distance = max(distances)
        else:
            max_distance = 0

        total_size = frame_count
        hist, _ = np.histogram(distances, bins=20, range=(0, max_distance))
        relative_frequencies = hist / total_size

        plt.bar(np.arange(20), relative_frequencies, color='blue', edgecolor='black', width=0.8)
        plt.ylim(0, 1)

        os.makedirs(f'{histogram_output_folder_path}/{class_name}/{case_num-1}/', exist_ok=True)
        plt.savefig(f'{histogram_output_folder_path}/{class_name}/{case_num-1}/{key}.png')
        plt.clf()
        
    
def segmentation_coordinations(frame):
    """
    A method for finding a bounding box (segmented area) information (x, y, w, h) out of black background.

    args:
    frame(image): an image frame with segmented area and black background. 
    
    return:
    a first frame image.
    a list of coordinates of the segmentation area.
    """
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #make the frame greyscale for easier binary threshold
    _, binary_image = cv.threshold(frame, 1, 255, cv.THRESH_BINARY)
    contours, _  = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if contours:
        segmented_box = max(contours, key=cv.contourArea)
        x,y,w,h = cv.boundingRect(segmented_box)
        return frame, [x, y, w, h]
    
def block_features(frame, coordinates, n_blocks):
    """
    A method for generating a block-based feature points to track during optical flow extraction.

    Args:
        frame (image): An image frame with segmentation.
        coordinates ([int]): A list of coordinates of the bounding box (segmentation area) in black background.
        n_blocks (int): Number of blocks to set tracking points.

    Returns:
        bool: Boolean value to check if the process was successful.
        np.ndarray: Numpy array containing feature point coordinates.
    """
    boo = False
    result = None  
    try:
        start_x, start_y, width, height = coordinates

        x = int(width / (n_blocks + 1))
        y = int(height / (n_blocks + 1))

        result = []
        for row in range(n_blocks):
            for col in range(n_blocks):
                box_point_x = int(start_x + (col + 1) * x)
                box_point_y = int(start_y + (row + 1) * y)
                result.append([np.float32(box_point_x), np.float32(box_point_y)])

        result = np.array(result)

        if result.shape[0] == n_blocks * n_blocks:
            result = result.reshape(n_blocks * n_blocks, 1, 2)
        else:
            raise ValueError("Unexpected number of feature points in result.")

        boo = True

    except Exception as e:
        print("Error at Block Feature Generation function:", e)

        debug_mode = True
        if debug_mode:
            cv.imshow("error", frame)
            cv.waitKey(0)
            cv.destroyAllWindows()

    finally:
        return boo, result
                    
def extract_optical_flow(input_folder, output_folder, class_name, size=None, threshold=2):
    """
    A method for processing a video to draw/track optical flow. 
    """
    os.makedirs(output_folder, exist_ok=True)

    if(size is None):
        size = len(os.listdir(input_folder))

    #Lukas-kanade optical flow parameters
    lk_params = dict(winSize=(50,50), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    color = (0, 255, 0)
    dictionary = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]} #directions 0 - 8 which is top, right top, ... etc.
    frame_count = 0
    num_videos_processed = 0
    case_num = 0

    while num_videos_processed < size:
        video_list = os.listdir(input_folder)
        for video_file_name in video_list:
            video_file_path = os.path.join(input_folder, video_file_name)
            cap = cv.VideoCapture(video_file_path)

            if not cap.isOpened():
                print(f"Error: Could not open {video_file_path}")
                case_num = case_num - 1
                continue

            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            output_video_path = os.path.join(output_folder, f"optical_flow_{video_file_name}")
            output_writer = cv.VideoWriter(output_video_path, fourcc, 20.0, (640, 480))

            ok, frame = cap.read()

            if np.all(frame==0):
                print("frame skipped due to mal-detection. No hand segmentation was happened on this frame.")
                continue

            # prev_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            
            first_grayscale_frame, coordinates = segmentation_coordinations(frame)

            boo, first_feature_points = block_features(frame, coordinates, 6)

            mask = np.zeros_like(frame)

            while cap.isOpened():
                ok, frame = cap.read()
                
                if not ok:
                    break
                    
                frame_count += 1
                
                if np.all(frame==0):
                    prinit("Blank frame is detected")
                    continue
                    
                cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
                next_grayscale_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                # calculates sparse optical flow by Lucas-Kanade method
                next_feature_points, status, error = cv.calcOpticalFlowPyrLK(first_grayscale_frame, next_grayscale_frame, first_feature_points, None, **lk_params)
                good_old = first_feature_points[status==1]
                good_new = next_feature_points[status==1]

                direction_detector = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0}
                sub_dictionary = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}

                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    new_x, new_y = new.ravel()
                    old_x, old_y = old.ravel()

                    displacement, direction = calculate_direction((old_x, old_y), (new_x, new_y))

                    displacement = eliminate_displacement(displacement, threshold)

                    if displacement != 0: 
                        direction_detector[direction] = direction_detector[direction]+1
                        sub_dictionary[direction].append(displacement)

                    mask_with_flow = cv.line(mask, (int(new_x), int(new_y)), (int(old_x),int(old_y)), color, 1)
                    frame = cv.circle(frame, (int(new_x), int(new_y)), 1, color, -1)

                if not check_fret_movement(direction_detector[2], direction_detector[6]):
                    for key, values in sub_dictionary.items():
                        for element in values:
                            dictionary[key].append(element)

                #update frame and feature points to next frame info
                first_grayscale_frame = next_grayscale_frame.copy()
                first_feature_points = good_new.reshape(-1, 1, 2)

                #write processed frame 
                output_frame = cv.add(frame, mask)
                output_writer.write(output_frame)

            # Release the resources
            cap.release()
            output_writer.release()
            cv.destroyAllWindows()

            print(f"Sparse optical flow video saved: {output_video_path}")
                    
            num_videos_processed += 1
            case_num += 1

            # create histogram for each direction
            print("Iteration at: ", class_name)
            print("Current Number: ",num_videos_processed)
            print("case number: ", case_num)
            create_histgram_of_motion(dictionary, class_name, case_num, frame_count)

            # Initialize dictionary and frame count for next video
            dictionary = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
            frame_count = 0  

def main():
    
    with open(BASE_DIR / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    input_folder_path = BASE_DIR / config["segmentation_output_data_path"]
    output_folder_path = BASE_DIR / config["optical_flow_output_data_path"]
    
    for folder in os.listdir(input_folder_path):
        if(folder != "norm"):
            input_folder = os.path.join(input_folder_path, folder)
            output_folder = os.path.join(output_folder_path, folder)
            extract_optical_flow(input_folder, output_folder, folder)

if __name__ == '__main__':
    main()
