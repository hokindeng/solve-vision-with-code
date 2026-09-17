import cv2
import numpy as np
import os
import subprocess

def main():
    img_path = '/app/first_frame.png'
    out_path = '/app/output/video.mp4'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read {img_path}")
        
    # Find green border to identify the moving object
    green_mask = cv2.inRange(img, (0, 200, 0), (50, 255, 50))
    y_green, x_green = np.where(green_mask > 0)
    gx_min, gx_max = x_green.min(), x_green.max()
    gy_min, gy_max = y_green.min(), y_green.max()
    gw = gx_max - gx_min + 1
    gh = gy_max - gy_min + 1
    
    # Calculate the center of the moving object (green border)
    source_cx = (gx_min + gx_max) / 2.0
    
    # Find red star to identify the target position
    red_mask = cv2.inRange(img, (0, 0, 200), (50, 50, 255))
    y_red, x_red = np.where(red_mask > 0)
    
    # Target center is the center of the red star (which is at the center of the target object)
    target_cx = (x_red.min() + x_red.max()) / 2.0
    
    # Total distance to move horizontally
    total_dx = int(round(target_cx - source_cx))
    
    # Extract the patch (moving object + green border)
    patch = img[gy_min:gy_max+1, gx_min:gx_max+1].copy()
    
    # We use a mask for pasting to be safe, though the background is white.
    # Any pixel that is not perfectly white will be considered part of the object.
    patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    _, patch_mask = cv2.threshold(patch_gray, 254, 255, cv2.THRESH_BINARY_INV)
    
    frames_dir = '/app/frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    num_frames = 60
    for i in range(num_frames):
        # Progress from 0.0 to 1.0
        progress = i / (num_frames - 1)
        dx = int(round(progress * total_dx))
        
        frame = img.copy()
        
        # Erase the object from its original position by filling with white background
        frame[gy_min:gy_max+1, gx_min:gx_max+1] = (255, 255, 255)
        
        # Calculate new position
        new_x_min = gx_min + dx
        new_x_max = gx_max + dx
        
        # Region of interest in the new position
        roi = frame[gy_min:gy_max+1, new_x_min:new_x_max+1]
        
        # Black out the object area in ROI
        roi_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(patch_mask))
        # Extract the object from patch
        roi_fg = cv2.bitwise_and(patch, patch, mask=patch_mask)
        # Combine them
        frame[gy_min:gy_max+1, new_x_min:new_x_max+1] = cv2.add(roi_bg, roi_fg)
        
        frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
        cv2.imwrite(frame_path, frame)
        
    # Compile video using ffmpeg
    cmd = [
        'ffmpeg', '-y',
        '-framerate', '16',
        '-i', os.path.join(frames_dir, 'frame_%04d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        out_path
    ]
    subprocess.run(cmd, check=True)
    
if __name__ == '__main__':
    main()
