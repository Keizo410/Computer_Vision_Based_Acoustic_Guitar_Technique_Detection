import cv2
import os

def checkWindowSize(video_path):
    
    cap = cv2.VideoCapture(video_path) #0 for camera

    if cap.isOpened(): 
        width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
        fps = cap.get(cv2.CAP_PROP_FPS) # float `fps`
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) # float `total_frame_in_the_video` (should not be applicable for camera)
       
def play_video_frame(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    print("Playing: ",video_path)

    # Initialize frame index
    frame_index = 0

    while True:
        # Set the frame position to the current index
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # Read a frame from the video
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret:
            break

        # Display the frame
        cv2.imshow("Video", frame)

        # Wait for a key press
        key = cv2.waitKey(0)

        # Move to the next frame if 'n' is pressed
        if key & 0xFF == ord('n'):
            frame_index += 1

        # Move to the previous frame if 'p' is pressed
        elif key & 0xFF == ord('p'):
            frame_index -= 1

            # Ensure frame index does not go below 0
            if frame_index < 0:
                frame_index = 0

        # Break the loop if 'q' is pressed
        elif key & 0xFF == ord('q'):
            break

    # Release the video capture object and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()


def play_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)    # Check if the video opened successfully
    print("Playing: ",video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Read and display each frame of the video
    while True:
        ret, frame = cap.read()  # Read a frame from the video
        # Check if the frame was read successfully
        if not ret:
            break
        width = 640
        height = 480
        dim = (width, height)

        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        # frame = cv2.resize(frame, (640, 480))
        # frame = cv2.flip(frame, -1)
        
        # Display the frame 
        cv2.imshow("Video", frame)

         # Wait for a key press; break the loop if 'q' is pressed
        # key = cv2.waitKey(0)
        # if key & 0xFF == ord('q'):
        #     break

        # # Wait for a key press; break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()
    
def play_all(video_path):
    # # # Iterate over all files in the directory
    for filename in os.listdir(video_path):
        # Check if the file is an MP4 video
        if filename.endswith(".mp4") or filename.endswith("MP4"):
            # Construct the full path to the video file
            full_path = os.path.join(video_path, filename)
            # Call the function to play the video
            checkWindowSize(full_path)
            play_video(full_path)
            # play_video_frame(full_path)


def main():
    
    # Path to the video directory
    left = ["../actions_dataset/data/acc_box/each_hands/lk_op/left/norm",
        "../actions_dataset/data/acc_box/each_hands/lk_op/left/ham",
        "../actions_dataset/data/acc_box/each_hands/lk_op/left/pull",
        "../actions_dataset/data/acc_box/each_hands/lk_op/left/slide",
        "../actions_dataset/data/acc_box/each_hands/left/pull",
        ]
    
    right = ["../actions_dataset/data/acc_box/each_hands/lk_op/right/norm",
        "../actions_dataset/data/acc_box/each_hands/lk_op/right/ham",
        "../actions_dataset/data/acc_box/each_hands/lk_op/right/pull",
        "../actions_dataset/data/acc_box/each_hands/lk_op/right/slide" ]

    # video_path = left[6]
    video_path = "..\\..\\..\\guitar_technique_detection\\data\\actions_dataset\\data\\original\\ham"
    play_all(video_path)

main()
