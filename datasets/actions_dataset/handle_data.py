import cv2 
import moviepy.editor as mp
import os

# Check image height and width
def resize(folder_path, output_folder):
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    files = os.listdir(folder_path)
    for file in files: 
            try:
                full_path = os.path.join(folder_path, file)  # Get the full path to the file
                vid = cv2.VideoCapture(full_path)
                height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
                width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)

                if height != 480 or width != 640:
                    clip = mp.VideoFileClip(full_path)
                    clip_resized = clip.resize((640,480))  # Resize to a height of 640px
                    new_file = os.path.join(output_folder, file)
                    clip_resized.write_videofile(new_file)
                    vid = cv2.VideoCapture(new_file)
                    resized_height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    resized_width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
                    print("Resized video dimensions:", resized_height, "x", resized_width)
            except cv2.error as error:
                print("cv2 error:", error)
            except Exception as e:
                print("Error occurred at:", file, "Error:", e)

import cv2 as cv
import os

def split_video(input_folder, output_folder, frames_per_segment=32):
    # Iterate over each video file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):
            input_video_path = os.path.join(input_folder, filename)
            output_video_folder = os.path.join(output_folder, os.path.splitext(filename)[0])
            
            # Call split_video_single function to split each video file
            split_video_single(input_video_path, output_video_folder, frames_per_segment)

def split_video_single(input_video_path, output_folder, frames_per_segment=32):
    # Open the input video file
    cap = cv.VideoCapture(input_video_path)

    # Check if the input video file was opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open input video file {input_video_path}.")
        return

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    # Calculate the number of segments
    num_segments = total_frames // frames_per_segment

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over segments
    for i in range(num_segments):
        # Set the start and end frame for the segment
        start_frame = i * frames_per_segment
        end_frame = min((i + 1) * frames_per_segment, total_frames)

        # Set the output file path
        output_video_path = os.path.join(output_folder, f"segment_{i+1}.mp4")

        # Create VideoWriter object for the segment
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(output_video_path, fourcc, 30, (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

        # Read and write frames for the segment
        cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
        for frame_num in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        # Release VideoWriter object
        out.release()

    # Release input video object
    cap.release()


import shutil

    
def move_and_rename_video(video_folder, parent_output_folder):
    # Iterate over subdirectories in the video folder
    for subdir in os.listdir(video_folder):
        subdir_path = os.path.join(video_folder, subdir)
        if os.path.isdir(subdir_path):
            # Iterate over files in the subdirectory
            for idx, filename in enumerate(os.listdir(subdir_path), start=21):
                if filename.endswith(".mp4"):
                    # Construct the source and destination paths
                    src_file_path = os.path.join(subdir_path, filename)
                    new_filename = f"norm{idx}.mp4"
                    dest_file_path = os.path.join(parent_output_folder, new_filename)

                    # Move the file to the parent output folder
                    shutil.move(src_file_path, dest_file_path)


def main():
    input_folder = "./data/acc_box/resized/both_hands/"
    # output_folder = "./data/acc_box/resized/both_hands/norm/output_segments"
    output_folder = "./data/acc_box/resized/both_hands/"

    # move_and_rename_video(output_folder, input_folder)
    resize(input_folder, output_folder)

if __name__ == '__main__':
    main()
