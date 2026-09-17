import cv2
import numpy as np
import os
import subprocess

def main():
    # Load the first frame
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("Could not load /app/first_frame.png")

    # We will generate 50 frames
    num_frames = 50
    fps = 16
    
    # Target positions
    # The upper line is at y=459..462. We want to start just below it, so top of line at y=463.
    # For thickness 4, center of endpoint should be 465.
    # The lower line is at y=663..666. We want to end just above it, so bottom of line at y=662.
    # For thickness 4, center of endpoint should be 660.
    x_center = 793
    y_start = 465
    y_end = 660
    
    # Prepare directory for frames
    os.makedirs('/app/frames', exist_ok=True)
    
    for i in range(num_frames):
        frame = img.copy()
        
        if i > 0:
            current_y = y_start + (y_end - y_start) * i / (num_frames - 1)
            # Draw the red line
            cv2.line(frame, (x_center, y_start), (x_center, int(round(current_y))), (0, 0, 255), 4)
            
        cv2.imwrite(f'/app/frames/frame_{i:04d}.png', frame)
        
    # Ensure output directory exists
    os.makedirs('/app/output', exist_ok=True)
    
    # Use ffmpeg to encode the video
    ffmpeg_cmd = [
        'ffmpeg',
        '-y', # Overwrite output
        '-framerate', str(fps),
        '-i', '/app/frames/frame_%04d.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '/app/output/video.mp4'
    ]
    
    subprocess.run(ffmpeg_cmd, check=True)

if __name__ == '__main__':
    main()
