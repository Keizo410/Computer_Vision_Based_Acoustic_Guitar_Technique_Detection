import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

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

#block number should be able to create n x n blocks
def block_featrues(frame, seg_info, n):
    boo = False 
    try:
        # frame_info should contain [x , y, w, h]. This is segmented part info from frame. Then calculate the blocks withing this segment
        x = int(seg_info[2]/(n+1)) # 1 block width
        y = int(seg_info[3]/(n+1)) # 1 block height
        sx = seg_info[0]    # starting point of segment area
        sy = seg_info[1]    # starting point of segment area

        array =[]
        for h in range(n):
            for v in range(n):
                a = (v+1)*x  #width
                b = (h+1)*y  #height
                a = int(sx + a)
                b = int(sy + b)
                array.append([np.float32(a), np.float32(b)])

        array = np.array(array)
        array = array.reshape(n*n, 1, 2)

        boo = True

    except Exception as e : 
        boo = False 
        print(e)
        cv2.imshow("error", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    finally:
        return boo, array

def seg_coordination(frame):
    # Threshold the frame to create a binary mask
    _, thresh = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
    # Find the bounding box of the largest contour
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        return [x, y, w, h]

def create_histgram_of_motion(dictionary, class_name, size = 0, case_num = None, frame_count = None):
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
    for key, distances in dictionary.items():
    
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
        # plt.xlabel('Distance')
        # plt.ylabel('Relative Frequency')
        # plt.title(f'Histogram for Distance (Direction {key})')

        if not os.path.exists(f'../../../guitar_technique_detection/data/histgram_of_motion_dataset/{class_name}/{case_num-1}/'):
            os.makedirs(f'../../../guitar_technique_detection/data/histgram_of_motion_dataset/{class_name}/{case_num-1}/')

        plt.savefig(f'../../../guitar_technique_detection/data/histgram_of_motion_dataset/{class_name}/{case_num-1}/{key}.png')
        plt.clf()
    
def extract_optical_flow(input_folder, output_folder, class_name, size = None, threshold = 2):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # check there is size specification: if there is not specification, then iterate all of the files in the folder
    if(size is None):
        size = len(os.listdir(input_folder))

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
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            out_video_path = os.path.join(output_folder, f"optical_flow_{video_file}")
            out = cv2.VideoWriter(out_video_path, fourcc, 20.0, (640, 480))

            ok, frame = cap.read()

            # Check if the frame is all black
            if np.all(frame == 0):
                print("here")
                continue  # Skip this frame
                    
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #get block based points to track
            # prev = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
            coord = seg_coordination(prev_gray)
            
            boo, prev = block_featrues(frame, coord, 6)

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
                    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color, 1)
                    # Draws filled circle (thickness of -1) at new position with green color and radius of 3
                    frame = cv2.circle(frame, (int(a), int(b)), 1, color, -1)

                if(direction_detector[2] < 18 or direction_detector[6] < 18 ):
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
            create_histgram_of_motion(dictionary, class_name, size, case_num, frame_count)

            dictionary = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
            direction = []
            displacement = []
            frame_count = 0  # Initialize frame count    

def main():
    input_folder_path = "../../../guitar_technique_detection/data/actions_dataset/data/segmented"
    output_folder_path = "../../../guitar_technique_detection/data/actions_dataset/data/optical_flow"

    # iterate through each technique folders
    for folder in os.listdir(input_folder_path):
        if(folder != "norm"):
            input_folder = os.path.join(input_folder_path, folder)
            output_folder = os.path.join(output_folder_path, folder)
            extract_optical_flow(input_folder, output_folder, folder)

if __name__ == '__main__':
    main()

