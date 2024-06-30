import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# # def finger_tip_feature(detector, img):
#     # Convert BGR image to RGB
#     frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

#     # Detect hand landmarks from the input image.
#     detection_result = detector.detect(image)
#     hand_landmarks_list = detection_result.hand_landmarks
#     indices_to_keep = [4, 8, 12, 16, 20]

#     # Create lists to store x, y, and z coordinates of each landmark
#     x_coords, y_coords, z_coords = [], [], []
#     array = []

#     # Loop through the detected hands
#     for hand_landmarks in hand_landmarks_list:
#         # Create a NormalizedLandmarkList protobuf message
#         hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#         hand_landmarks_proto.landmark.extend([
#             landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
#             for i, landmark in enumerate(hand_landmarks)
#             if i in indices_to_keep
#         ])
        
#         # Extract and store x, y, and z coordinates of each landmark
#         for landmark in hand_landmarks_proto.landmark:
#             x_coords.append(landmark.x)
#             y_coords.append(landmark.y)
#             z_coords.append(landmark.z)
#             array.append([np.float32(landmark.x), np.float32(landmark.y)])
          
#         array = np.array(array)
#         array = array.reshape(5, 1,2)

#     return array
def finger_tip_feature(detector, img):
    # Convert BGR image to RGB
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Detect hand landmarks from the input image.
    detection_result = detector.detect(image)
    hand_landmarks_list = detection_result.hand_landmarks
    indices_to_keep = [5,6,7, 8,9,10,11, 12,13,14,15, 16,17,18, 19, 20]

    # Create lists to store x, y, and z coordinates of each landmark
    x_coords, y_coords, z_coords = [], [], []

    # Loop through the detected hands
    for hand_landmarks in hand_landmarks_list:
        # Create a NormalizedLandmarkList protobuf message
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for i, landmark in enumerate(hand_landmarks)
            if i in indices_to_keep
        ])
        
        # Extract and store x, y, and z coordinates of each landmark
        for landmark in hand_landmarks_proto.landmark:
            x_coords.append(landmark.x)
            y_coords.append(landmark.y)
          
    # Convert lists to numpy arrays
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    
    # Denormalize coordinates based on image width and height
    img_height, img_width = img.shape[:2]
    x_coords = np.float32((x_coords * img_width).astype(int))
    y_coords = np.float32((y_coords * img_height).astype(int))
    # Stack x and y coordinates into a single array
    array = np.column_stack((x_coords, y_coords))
    # z_array = np.column_stack((z_coords))
    array = array.reshape(16, 1, 2)
    
    return array

# def create_point_images(x_coords, y_coords, img_path, output_folder):
#     # Read the input image
#     img = cv2.imread(img_path)

#     # Iterate through the coordinates and draw points on the image
#     for x, y in zip(x_coords, y_coords):
#         # Convert the coordinates to integer
#         x_int, y_int = int(x * img.shape[1]), int(y * img.shape[0])  # Convert normalized coordinates to image coordinates
#         # Draw a point on the image
#         cv2.circle(img, (x_int, y_int), radius=5, color=(0, 255, 0), thickness=-1)

#     # Save the image with points
#     output_path = output_folder + "/point_image.jpg"
#     cv2.imwrite(output_path, img)

#     print("Point image saved successfully at:", output_path)

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import math 

def calculate_direction(old_point, new_point):
    """
    Calculate the direction from an old point to a new point.

    :param old_point: Coordinates of the old point as a tuple (x, y).
    :param new_point: Coordinates of the new point as a tuple (x, y).
    :return: Angle (in radians) representing the direction from the old point to the new point.
    """
    # Compute the displacement vector from the old point to the new point
    displacement = np.array(new_point) - np.array(old_point)

    # Calculate the distance between the points
    distance = np.linalg.norm(displacement)

    # If there's no movement (distance is close to zero), return the "no movement" sector index
    if distance < 0.01:
        return distance, 8

    # Compute the angle (in radians) between the displacement vector and the x-axis
    direction_rad = np.arctan2(displacement[1], displacement[0])

    # Convert the angle to degrees
    direction_deg = np.degrees(direction_rad)

    # Ensure the angle is positive
    if direction_deg < 0:
        direction_deg += 360

    # Calculate the direction sector index (0 to 7)
    direction_sector = int(direction_deg / 45)

    # print("displacement: ", distance)
    # print("degree: ", direction_sector)

    return distance, direction_sector



def createHistogramOfMotion_magnitude(dictionary, class_name, size = 0, case_num = None, frame_count = None):
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
    """
    if(size is not None):
        for key, distances in dictionary.items():
        
            # print(f"total size for direction {key} is: ",len(distances))
            if(len(distances)>0):
                # print("max distance: ", max(distances))
                max_distance = max(distances)
            else:
                max_distance = 0

            # total_size=len(distances)
            total_size = frame_count
            # Calculate relative frequencies
            hist, _ = np.histogram(distances, bins=20, range=(0, max_distance))
            relative_frequencies = hist / total_size

            # Plot relative frequencies
            plt.bar(np.arange(20), relative_frequencies, color='blue', edgecolor='black', width=0.8)

            plt.ylim(0, 1)

            # Set labels and title
            plt.xlabel('Distance')
            plt.ylabel('Relative Frequency')
            plt.title(f'Histogram for Distance (Direction {key})')
    
            if not os.path.exists(f'./images/hom/separate/all_videos/left/{class_name}/case_{case_num-1}/'):
                os.makedirs(f'./images/hom/separate/all_videos/left/{class_name}/case_{case_num-1}/')

            # Save the plot to a file (e.g., PNG, PDF, etc.)
            plt.savefig(f'./images/hom/separate/all_videos/left/{class_name}/case_{case_num-1}/dis_block_oplk_left_hand_{class_name}_{key}.png')
            # plt.savefig(f'./images/hom/separate/left_3/{class_name}/case_{case_num-1}/dis_block_oplk_left_hand_{class_name}_{key}.png')
            plt.clf()
    else: 
        
        for key, distances in dictionary.items():

            print(f"total size for direction {key} is: ",len(distances))
            print("max distance: ", max(distances))

            total_size = frame_count
            # Calculate relative frequencies
            hist, _ = np.histogram(distances, bins= 20, range=(0, max(distances)))
            print("frame count: ",frame_count)
            print("hist: ", hist)
            relative_frequencies = hist / total_size

            # Plot relative frequencies (x, y)
            plt.bar(np.arange(20), relative_frequencies, color='blue', edgecolor='black', width=0.8)

            plt.ylim(0, 1)

            # Set labels and title
            plt.xlabel('Distance')
            plt.ylabel('Relative Frequency')
            plt.title(f'Histogram for Distance (Direction {key})')

            # Save the plot to a file (e.g., PNG, PDF, etc.)
            plt.savefig(f'./images/hom/separate/each_technique/left/{class_name}/dis_block_oplk_left_hand_{class_name}_{key}.png')
            plt.clf()

def z_direction_feature(detector, img): 
    # Convert BGR image to RGB
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Detect hand landmarks from the input image.
    detection_result = detector.detect(image)
    hand_landmarks_list = detection_result.hand_landmarks
    indices_to_keep = [5,6,7, 8,9,10,11, 12,13,14,15, 16,17,18, 19, 20]

    # Create lists to store x, y, and z coordinates of each landmark
    z_coords = []

    # Loop through the detected hands
    for hand_landmarks in hand_landmarks_list:
        # Create a NormalizedLandmarkList protobuf message
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for i, landmark in enumerate(hand_landmarks)
            if i in indices_to_keep
        ])
        
        # Extract and store x, y, and z coordinates of each landmark
        for landmark in hand_landmarks_proto.landmark:
            z_coords.append(landmark.z)
          
    # Convert lists to numpy arrays
    z_coords = np.array(z_coords)
    
    # Denormalize coordinates based on image width and height
    z_coords = np.float32(z_coords)
    
    return z_coords

def process_2d_list(lst):
    # Count positive and negative values for each index
    counts = {'positive': [0] * len(lst[0]), 'negative': [0] * len(lst[0])}
    for sub_lst in lst:
        for i, val in enumerate(sub_lst):
            if val > 0:
                counts['positive'][i] += 1
            elif val < 0:
                counts['negative'][i] += 1

    # Filter values based on counts
    result = []
    for sub_lst in lst:
        new_sub_lst = []
        for i, val in enumerate(sub_lst):
            if counts['positive'][i] > 1 or counts['negative'][i] > 1:
                new_sub_lst.append(val)
            else:
                new_sub_lst.append(0)
        result.append(new_sub_lst)

    return result
    
def plot_cumulative_z_coordinates(z_coordinates_list, class_name, num):
    # Create an empty transposed list
    transposed_list = []

    # Iterate through the original list and populate the transposed list
    for point in range(16):
        array = []
        for frame in range(len(z_coordinates_list)):
            if(z_coordinates_list[frame].size != 0):
                array.append(z_coordinates_list[frame][point])
            else:
                array.append(0)
        transposed_list.append(array)

    # array = []
    # for ind in range(len(transposed_list)):
    #     data=normalize_between_minus_one_and_one(transposed_list[ind])
    #     array.append(data)
    # processed_list = process_2d_list(array)    
    
    for point in range(len(z_coordinates_list)):
        for ind in range(len(transposed_list)):
            if((ind+1)%4==0):
                y= normalize_between_minus_one_and_one(transposed_list[ind])
                # y = processed_list[ind]
                x = [i for i in range(len(z_coordinates_list))]
                # plotting the points 
                plt.plot(x, y, label = f"point {ind}")
            
    # naming the x axis
    plt.xlabel('x - axis')
    # naming the y axis
    plt.ylabel('y - axis')
    plt.ylim(-1, 1)
    plt.legend()
    # function to show the plot
    plt.savefig(f"./images/z/{class_name}/{num}")
    plt.clf()


def sparse_optical_flow_some_video(input_folder, output_folder, class_name, detector, size = None, threshold = 2):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # check there is size specification: if there is not specification, then iterate all of the files in the folder
    if(size is None):
        size = len(os.listdir(input_folder))

    # parameter for Shi-Tomasi corner detection
    # feature_params = dict(maxCorners=50, qualityLevel=0.2, minDistance=1, blockSize=7)
    # parameter for Lucas-kanade optical flow
    lk_params = dict(winSize=(50, 50), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # variable for color to draw optical flow track
    color = (0, 255, 0)

    # direction array
    dictionary = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
    direction = []
    displacement = []
    frame_count = 0  # Initialize frame count

    # Initialize a variable to keep track of the number of processed videos
    num_videos_processed = 0

    # Initialize a variable to keep track of the video number
    case_num = 0

    # Initialize a cumulative array for z 
    cumulative_data = []

    # Loop through videos in input folder until the desired number is reached
    while num_videos_processed < size:

        # get list of files
        videos = os.listdir(input_folder)
        
        # Check if the number of processed videos has reached the desired size
        if num_videos_processed >= size:
            break

        # Loop through videos in input folder
        for video_file in videos:

            # Check if the number of processed videos has reached the desired size
            if num_videos_processed >= size:
                break

            # make

            video_path = os.path.join(input_folder, video_file)
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"Error: Could not open {video_file}")
                case_num = case_num - 1
                continue

            # Define codec and VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out_video_path = os.path.join(output_folder, f"sparse_optical_flow_{video_file}")
            out = cv2.VideoWriter(out_video_path, fourcc, 20.0, (640, 480))

            ok, frame = cap.read()

            # Check if the frame is all black
            if np.all(frame == 0):
                print("here")
                continue  # Skip this frame
                    
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            prev = finger_tip_feature(detector, frame)
            
            mask = np.zeros_like(frame)

            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break

                # Increment frame count
                frame_count += 1

                # Check if the frame is all black
                if np.all(frame == 0):
                    print("here")
                    continue  # Skip this frame

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # calculates sparse optical flow by Lucas-Kanade method
                next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)

                ######################## z axis modeling #################################################
                # z_array = z_direction_feature(detector, frame)
                # cumulative_data.append(z_array)
                ###########################################################################################
                # selects good feature points for previous position
                good_old = prev[status == 1]
                # selects good feature points for next position
                good_new = next[status == 1]

               
                direction_detector = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0}
                sub_dictionary = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
                # Draws the optical flow tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    # Returns a contiguous flattened array as (x, y) coordinates for new point
                    a, b = new.ravel()
                    # Returns a contiguous flattened array as (x, y) coordinates for old point
                    c, d = old.ravel()

                    # calculate the direction and save it to the array to make histogram
                    disp, dirct = calculate_direction((c,d), (a,b))

                    # set threshold to eliminate small vectors
                    if(disp< threshold):
                        disp = 0

                    # # not include zero
                    if(disp != 0):
                        direction_detector[dirct] = direction_detector[dirct]+1
                        sub_dictionary[dirct].append(disp)

                        # dictionary[dirct].append(disp)

                    # Draws line between new and old position with green color and 2 thickness
                    mask = cv2.line(mask, (a, b), (c, d), color, 1)
                    # Draws filled circle (thickness of -1) at new position with green color and radius of 3
                    frame = cv2.circle(frame, (a, b), 1, color, -1)
                    

                if(direction_detector[2] < 20 or direction_detector[1] < 20 or direction_detector[3] < 20 or direction_detector[6] < 20 ):
                    # print(max(direction_detector))
                    for key, value in sub_dictionary.items():
                        for element in value:
                            dictionary[key].append(element)

                # Overlays the optical flow tracks on the original frame
                output = cv2.add(frame, mask)

                # Updates previous frame
                prev_gray = gray.copy()
                # Updates previous good feature points
                prev = good_new.reshape(-1, 1, 2)

                # Write the frame to output video
                out.write(output)

                # cv2.imshow("sparse optical flow", output)
                # if cv2.waitKey(10) & 0xFF == ord("q"):
                #     break

            # Release resources
            cap.release()
            out.release()
            cv2.destroyAllWindows()

            print(f"Sparse optical flow video saved: {out_video_path}")

            num_videos_processed += 1
            case_num += 1

            # create histogram for each direction
            print("Iteration at: ", class_name)
            print("Current Number: ",num_videos_processed)
            print("case number: ", case_num)
            createHistogramOfMotion_magnitude(dictionary, class_name, size, case_num, frame_count )

            dictionary = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
            direction = []
            displacement = []
            frame_count = 0  # Initialize frame count 

            # plot_cumulative_z_coordinates(cumulative_data, class_name, case_num)
            cumulative_data = []

def normalize_between_minus_one_and_one(data):
    min_val = min(data)
    max_val = max(data)
    normalized_data = [2 * ((x - min_val) / (max_val - min_val)) - 1 for x in data]
    skip = 1
    for i in range(len(normalized_data)-skip):
        curr = normalized_data[i]
        next = normalized_data[i+skip]
        if((curr>0 and next<0) or (curr<0 and next>0)):
            normalized_data[i] = curr
        else:
            normalized_data[i] = 0
    return normalized_data


def check(input_folder, output_folder, class_name, detector, count, size = None, threshold = 5):
    
    # direction array
    dictionary = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
    direction = []
    displacement = []
    frame_count = 0  # Initialize frame count

    # Initialize a variable to keep track of the number of processed videos
    num_videos_processed = 0

    # Initialize a variable to keep track of the video number
    case_num = 0

    # Initialize a cumulative array for z 
    cumulative_data = []

    # Loop through videos in input folder until the desired number is reached
        
    video_path = input_folder
    cap = cv2.VideoCapture(video_path)

    ok, frame = cap.read()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        # Check if the frame is all black
        if np.all(frame == 0):
            print("here")
            continue  # Skip this frame

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        ######################## z axis modeling #################################################
        z_array = z_direction_feature(detector, frame)
        cumulative_data.append(z_array)
        ###########################################################################################
        
    # Release resources
    cap.release()
    # out.release()
    cv2.destroyAllWindows()

    # create histogram for each direction
    print("Iteration at: ", class_name)
    print("case number: ", count)
    dictionary = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
    direction = []
    displacement = []
    frame_count = 0  # Initialize frame count 
    plot_cumulative_z_coordinates(cumulative_data, class_name, count)




def main():
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1, min_hand_detection_confidence = 0.1, min_tracking_confidence = 0.3 )
    detector = vision.HandLandmarker.create_from_options(options)

    input_folder_path = "../actions_dataset/data/acc_box/each_hands/left/"
    output_folder_path = "../actions_dataset/data/acc_box/each_hands/lk_op/left/"
    # output_folder_path = "../actions_dataset/data/acc_box/each_hands/lk_op/left_3/"

    # iterate through each technique folders
    for folder in os.listdir(input_folder_path):
        if(folder != "norm"):
            input_folder = os.path.join(input_folder_path, folder)
            output_folder = os.path.join(output_folder_path, folder)
            # sparse_optical_flow_all_video(input_folder, output_folder, folder)
            sparse_optical_flow_some_video(input_folder, output_folder, folder, detector)

if __name__ == '__main__':
    main()



# base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
# options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1, min_hand_detection_confidence = 0.1, min_tracking_confidence = 0.3 )

# detector = vision.HandLandmarker.create_from_options(options)
# input_folder = "../actions_dataset/data/acc_box/each_hands/left/"
# output_folder = "./images/z/"

# for category in os.listdir(input_folder):
#     target = os.path.join(input_folder, category)
#     out = os.path.join(output_folder, category)
#     count = 0
#     for file in os.listdir(target):
#         f = os.path.join(target, file)
#         print("processing: ", f"{f}" )
#         check(f, out, category, detector, count, size = None, threshold=5)
#         count = count + 1
