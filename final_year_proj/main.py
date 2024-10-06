import tkinter as tk
from tkinter import filedialog, Menu, Button
from PIL import Image, ImageTk
import cv2
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

frame_count = 0  # Global counter for frame filenames

# Initialize figure and axes as None globally
fig = None
ax1 = None
ax2 = None
canvas = None

# Function to open an MP4 file and display the first frame
def open_file():
    file_path = filedialog.askopenfilename(title="Select an MP4 File",
                                           filetypes=[("MP4 files", "*.mp4")])
    
    
    # Extract video name
    global video_name
    video_name = os.path.splitext(os.path.basename(file_path))[0]

    if file_path:
        export_as_frames(file_path)
        
def select_folders():
    # Open a dialog to select a folder
    folder_selected = filedialog.askdirectory(title="Select Folder Containing Frames")

    if folder_selected:
        avg_brightness_list, ssim_data = read_in_frames_from_folder(folder_selected)
        folder_name = os.path.basename(folder_selected)
        display_plots(avg_brightness_list, ssim_data, folder_name)
        
# Function to create and display the graph
def display_plots(brightness_data, ssim_data, folder_name):
    global fig, ax1, ax2, canvas  # Access global figure, axes, and canvas

    if not brightness_data:
        return

    # Check if figure and axes already exist
    if fig is None or ax1 is None or ax2 is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5))  # Create two subplots: one for brightness and one for SSIM
        print("New figure with subplots created.")
        
        # Create a canvas to embed the figure in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.get_tk_widget().pack(pady=40, padx=40, side=tk.TOP, anchor='n', expand=True, fill=tk.BOTH)

    # Plot the brightness data on the first subplot (ax1)
    ax1.plot(range(len(brightness_data)), brightness_data, label=f"'{folder_name}' Brightness")
    
    # Set labels and title for the first plot (brightness)
    ax1.set_title("Average Frame Brightness")
    ax1.set_xlabel("Frame Number")
    ax1.set_ylabel("Average Brightness")
    
    # Add legend to differentiate between multiple plots
    ax1.legend()
    
    # Plot the SSIM data on the second subplot (ax2)
    if ssim_data:
        ax2.plot(range(len(ssim_data)), ssim_data, label=f"'{folder_name}' SSIM")
        
        # Set labels and title for the second plot (SSIM)
        ax2.set_title("SSIM Between Consecutive Frames")
        ax2.set_xlabel("Frame Number")
        ax2.set_ylabel("SSIM Value")
        
        # Add legend to the second plot
        ax2.legend()
        
    # Adjust layout automatically
    plt.tight_layout()

    # Redraw the canvas to update the plots
    if canvas:
        canvas.draw()

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
    
def calculate_ssim(imageA, imageB):
    # Ensure that images are in floating-point format and normalized
    imageA = imageA.astype(np.float32) / 255.0
    imageB = imageB.astype(np.float32) / 255.0

    # Calculate means
    muA = cv2.GaussianBlur(imageA, (11, 11), 1.5)
    muB = cv2.GaussianBlur(imageB, (11, 11), 1.5)

    # Calculate variances and covariance
    sigmaA_sq = cv2.GaussianBlur(imageA * imageA, (11, 11), 1.5)
    sigmaB_sq = cv2.GaussianBlur(imageB * imageB, (11, 11), 1.5)
    sigmaAB = cv2.GaussianBlur(imageA * imageB, (11, 11), 1.5)

    sigmaA_sq -= muA * muA
    sigmaB_sq -= muB * muB
    sigmaAB -= muA * muB

    # Constants for SSIM computation (for normalized [0, 1] images)
    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)

    # Calculate SSIM
    ssim_map = ((2 * muA * muB + C1) * (2 * sigmaAB + C2)) / \
               ((muA * muA + muB * muB + C1) * (sigmaA_sq + sigmaB_sq + C2))

    return cv2.mean(ssim_map)[0]
    
def read_in_frames_from_folder(folder_path):
    
    # Get a list of all JPG files in the folder
    frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg")])
    
    # Create new array to store frames
    frame_array = []
    avg_brightness_list = []  # To store brightness for each frame
    ssim_list = [] # To store SSIM values
    
    # Process each frame
    for frame_file in frame_files:
        # Construct the full path to the frame
        frame_path = os.path.join(folder_path, frame_file)
        
        # Read the frame using OpenCV
        frame = cv2.imread(frame_path)
        frame_array.append(frame)
        
    prev_frame_grey = None # Store the previous frame in cycle
    for i, frame in enumerate(frame_array):
        # Convert frame to greyscale
        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate average brightness
        avg_brightness = np.mean(grey_frame)
        avg_brightness_list.append(avg_brightness)
        
        # Calculate SSIM with the previous frame (skip SSIM for the first frame)
        if prev_frame_grey is not None:
            ssim_value = calculate_ssim(prev_frame_grey, grey_frame)
            ssim_list.append(ssim_value)
        else:
            # For the first frame, SSIM is not applicable
            ssim_list.append(None)  # Set to None for the first frame
        
        # Update the previous frame
        prev_frame_grey = grey_frame
        
    return avg_brightness_list, ssim_list
        
        
# at this point a frame array exists and I can start plotting its info
    
        
# Create the main window
root = tk.Tk()
root.title("MP4 Frame Display")
root.geometry("600x400")

# Create a menubar
menu_bar = Menu(root)

# Create an exit button
exit_button = Button(root, text="exit", command=root.quit)
exit_button.pack(side=tk.BOTTOM, anchor='w', padx=10, pady=10)

# Create a file open button
export_video_button = Button(root, text="Export Video", command=open_file)
export_video_button.pack(side=tk.BOTTOM, anchor='w', padx=10, pady=10)

# Create an read frames button
read_frames_button = Button(root, text="Read Frames", command=select_folders)
read_frames_button.pack(side=tk.BOTTOM, anchor='w', padx=10, pady=10)


# Set the menubar to the window
root.config(menu=menu_bar)

# Create a label to display the image
panel = tk.Label(root)
panel.pack(padx=10, pady=10)


# Start the main loop
root.mainloop()
