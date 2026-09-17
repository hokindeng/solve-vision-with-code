import cv2
import numpy as np
import subprocess
from scipy.interpolate import CubicSpline
import os

def create_video():
    img_path = '/app/first_frame.png'
    output_video_path = '/app/output/video.mp4'
    frames_dir = '/app/frames'
    
    img = cv2.imread(img_path)
    if img is None:
        raise Exception(f"Could not read {img_path}")

    # Find platforms
    # Platform outline color: [120, 120, 0] (BGR)
    plat_mask = cv2.inRange(img, np.array([120, 120, 0]), np.array([120, 120, 0]))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(plat_mask)
    platforms = []
    for i in range(1, num_labels):
        platforms.append(centroids[i])
    # Sort platforms from left to right
    platforms.sort(key=lambda p: p[0])

    # Find ball
    # Ball outline: [67, 0, 175], fill: [147, 20, 255]
    c1, c2 = np.array([67, 0, 175]), np.array([147, 20, 255])
    ball_mask = ((img == c1).all(axis=2) | (img == c2).all(axis=2))
    coords = np.argwhere(ball_mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Extract ball patch and mask
    ball_patch = img[y_min:y_max+1, x_min:x_max+1].copy()
    ball_patch_mask = ball_mask[y_min:y_max+1, x_min:x_max+1].copy()

    ball_center_x = (x_min + x_max) / 2.0
    ball_bottom_y = y_max

    # Time parameters for splines (based on distances)
    t_pts = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    pts_x = [ball_center_x] + [p[0] for p in platforms]
    pts_y = [ball_bottom_y] + [p[1] for p in platforms]
    
    # Cubic splines for x and y trajectory
    cs_x = CubicSpline(t_pts, pts_x)
    cs_y = CubicSpline(t_pts, pts_y)

    # Clean background by removing the ball
    bg = img.copy()
    bg[ball_mask] = [255, 255, 255]

    # Video writer setup
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    
    frames_count = 64
    for f in range(frames_count):
        # Smoothstep easing
        x_progress = f / (frames_count - 1)
        progress = x_progress * x_progress * (3 - 2 * x_progress)
        t = progress * 13
        
        x = float(cs_x(t))
        y = float(cs_y(t))
        
        h, w = ball_patch.shape[:2]
        
        # Calculate destination box coordinates
        target_x_min = int(round(x - w / 2.0))
        target_x_max = target_x_min + w
        target_y_max = int(round(y)) + 1
        target_y_min = target_y_max - h
        
        frame = bg.copy()
        
        # Clip to frame boundaries
        y_start = max(0, target_y_min)
        y_end = min(frame.shape[0], target_y_max)
        x_start = max(0, target_x_min)
        x_end = min(frame.shape[1], target_x_max)
        
        patch_y_start = y_start - target_y_min
        patch_y_end = h - (target_y_max - y_end)
        patch_x_start = x_start - target_x_min
        patch_x_end = w - (target_x_max - x_end)
        
        # Draw the ball patch
        for c in range(3):
            frame[y_start:y_end, x_start:x_end, c] = np.where(
                ball_patch_mask[patch_y_start:patch_y_end, patch_x_start:patch_x_end],
                ball_patch[patch_y_start:patch_y_end, patch_x_start:patch_x_end, c],
                frame[y_start:y_end, x_start:x_end, c]
            )
            
        cv2.imwrite(f'{frames_dir}/frame_{f:04d}.png', frame)
        
    # Combine frames into an mp4 video using ffmpeg
    cmd = [
        'ffmpeg', '-y', '-framerate', '16', '-i', f'{frames_dir}/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', output_video_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Video generated at {output_video_path}")

if __name__ == '__main__':
    create_video()
