import tkinter as tk
from tkinter import filedialog, Menu, Button
from PIL import Image, ImageTk
import cv2
import os
from tqdm import tqdm

frame_count = 0  # Global counter for frame filenames


# Function to open an MP4 file and display the first frame
def open_file():
    file_path = filedialog.askopenfilename(title="Select an MP4 File",
                                           filetypes=[("MP4 files", "*.mp4")])
    
    
    # Extract video name
    global video_name
    video_name = os.path.splitext(os.path.basename(file_path))[0]

    if file_path:
        export_as_frames(file_path)

def export_as_frames(video_filepath):
    global frame_count

    # Create a folder with the same name as the video if it doesn't exist
    if not os.path.exists(video_name):
        os.makedirs(video_name)
        
    # Open the video file
    cap = cv2.VideoCapture(video_filepath)
    
    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Loop through all frames
    for frame_count in tqdm(range(total_frames), desc="Extracting frames"):
        ret, frame = cap.read()
        if not ret:
            break  # Stop if there are no more frames

        # Save the frame as a JPG file in the video-named folder
        frame_filename = f"{video_name}/frame_{frame_count:04d}.jpg"  # e.g., video_name/frame_0001.jpg
        cv2.imwrite(frame_filename, frame)  # Save using OpenCV

    # Release the video capture object
    cap.release()
    print(f"Frames extracted and saved to '{video_name}'.")


# Create the main window
root = tk.Tk()
root.title("MP4 Frame Display")
root.geometry("600x400")

# Create a menubar
menu_bar = Menu(root)

# Create a file open button
open_file_button = Button(root, text="Export Video", command=open_file)
open_file_button.pack(side=tk.LEFT, anchor='s', padx=10, pady=10)

# Create an exit button
exit_button = Button(root, text="exit", command=root.quit)
exit_button.pack(side=tk.RIGHT, anchor='s', padx=10, pady=10)

# Set the menubar to the window
root.config(menu=menu_bar)

# Create a label to display the image
panel = tk.Label(root)
panel.pack(padx=10, pady=10)

# Start the main loop
root.mainloop()
