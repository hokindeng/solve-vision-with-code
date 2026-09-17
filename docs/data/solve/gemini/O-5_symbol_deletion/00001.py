import cv2
import numpy as np
import os
import subprocess
import tempfile
import shutil

def create_video():
    first_frame = cv2.imread('/app/first_frame.png')
    if first_frame is None:
        raise FileNotFoundError("Could not find /app/first_frame.png")
        
    h, w, c = first_frame.shape
    
    # Identify the red border and the symbol inside it
    red_mask = (first_frame[:,:,0] == 0) & (first_frame[:,:,1] == 0) & (first_frame[:,:,2] == 255)
    coords = np.argwhere(red_mask)
    if len(coords) == 0:
        print("No red border found!")
        return
        
    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)
    
    num_frames = 46
    fps = 16
    output_dir = '/app/output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Background color is white
    bg_color = np.array([255, 255, 255], dtype=float)
    
    # Target ROI in the first frame
    roi = first_frame[min_y:max_y+1, min_x:max_x+1].astype(float)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(num_frames):
            # Calculate alpha (1.0 at start, 0.0 at end)
            if num_frames > 1:
                alpha = 1.0 - (i / (num_frames - 1))
            else:
                alpha = 0.0
                
            frame = first_frame.copy()
            
            # Blend the ROI towards the background color
            blended_roi = roi * alpha + bg_color * (1.0 - alpha)
            frame[min_y:max_y+1, min_x:max_x+1] = np.round(blended_roi).astype(np.uint8)
            
            cv2.imwrite(os.path.join(temp_dir, f"frame_{i:04d}.png"), frame)
            
        # Create video using ffmpeg
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps), "-i", os.path.join(temp_dir, "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", f"{output_dir}/video.mp4"
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
if __name__ == "__main__":
    create_video()
