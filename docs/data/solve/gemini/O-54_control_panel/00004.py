import cv2
import numpy as np
import imageio
import os

def solve():
    input_path = '/app/first_frame.png'
    output_path = '/app/output/video.mp4'
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not read {input_path}")
        
    # We infer that Left = Red, Middle = Magenta.
    # Unit 2 is on the Left and its light is Red.
    # Units 1 and 3 are in the Middle and their lights are Magenta.
    # We need to move the levers of Unit 1 and 3 to the Left.
    
    # Reconstruct the clean track (background of the lever slot)
    # Unit 2 track has lever on the left, so middle and right marks are visible
    clean_track = img[620:734, 396:628].copy()
    # Erase the gray lever (replace with black)
    clean_track[np.all(clean_track == [128, 128, 128], axis=-1)] = [0, 0, 0]
    
    # Track 1 has lever in the middle, so left mark is visible
    track1 = img[620:734, 89:321]
    # Copy the left mark from track 1 (x < 60)
    left_mark_mask = np.all(track1[:, :60] == [240, 240, 240], axis=-1)
    clean_track[:, :60][left_mark_mask] = [240, 240, 240]

    # Initialize video writer (exactly 24 frames, 16 fps)
    writer = imageio.get_writer(
        output_path, 
        fps=16, 
        macro_block_size=None, 
        pixelformat='yuv420p',
        codec='libx264'
    )

    for i in range(24):
        t = i / 23.0
        
        # Lever x position: from 87 (middle) to 10 (left)
        lever_x = int(round(87 * (1 - t) + 10 * t))
        
        # Light color B channel: from 255 (magenta) to 0 (red)
        B = int(round(255 * (1 - t) + 0 * t))
        
        frame = img.copy()
        
        # Replace magenta color with the interpolated color (to red)
        magenta_mask = np.all(frame == [255, 0, 255], axis=-1)
        frame[magenta_mask] = [B, 0, 255]
        
        # Draw Unit 1 lever on its track
        t1 = clean_track.copy()
        t1[28:28+57, lever_x:lever_x+57] = [128, 128, 128]
        frame[620:734, 89:321] = t1
        
        # Draw Unit 3 lever on its track
        t3 = clean_track.copy()
        t3[28:28+57, lever_x:lever_x+57] = [128, 128, 128]
        frame[620:734, 703:935] = t3
        
        # OpenCV uses BGR, imageio uses RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)

    writer.close()

if __name__ == '__main__':
    solve()
