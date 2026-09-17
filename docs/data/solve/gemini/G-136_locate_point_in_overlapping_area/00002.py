import cv2
import numpy as np
import os
import subprocess

def solve():
    # Load first frame
    img = cv2.imread('/app/first_frame.png')
    
    # 1. Identify the shapes and overlap
    colors, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
    valid_colors = []
    for c, count in zip(colors, counts):
        if count > 1000:
            if np.mean(c) > 240: continue # white background
            if np.mean(c) < 50: continue # black dots
            valid_colors.append((count, c))
    
    valid_colors.sort(reverse=True)
    if len(valid_colors) >= 3:
        overlap_color = valid_colors[2][1]
    else:
        # Fallback if somehow not exactly 3 colors
        overlap_color = valid_colors[-1][1]

    # Create mask for overlap
    mask_overlap = cv2.inRange(img, overlap_color - 2, overlap_color + 2)
    
    # Dilate/Close to fill holes
    kernel = np.ones((25, 25), np.uint8)
    closed_overlap = cv2.morphologyEx(mask_overlap, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed_overlap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlap_cnt = max(contours, key=cv2.contourArea)
    
    # 2. Identify the dots
    black_mask = cv2.inRange(img, np.array([0,0,0]), np.array([50,50,50]))
    points_contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    inside_points = []
    for c in points_contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            dist = cv2.pointPolygonTest(overlap_cnt, (cX, cY), True)
            # Distance should be positive and > some small threshold to not be on boundary
            if dist > 6: 
                inside_points.append((cX, cY))
                
    # 3. Generate frames
    os.makedirs('/app/frames', exist_ok=True)
    num_frames = 37
    
    for i in range(num_frames):
        frame = img.copy()
        
        # Calculate angle for animation (0 to 360)
        # Frame 0: angle 0 (no circle)
        # Frame 36: angle 360 (full circle)
        angle = int((i / (num_frames - 1)) * 360)
        
        if angle > 0:
            for pt in inside_points:
                cv2.ellipse(frame, pt, (25, 25), 0, 0, angle, (0, 0, 255), 4, cv2.LINE_AA)
                
        cv2.imwrite(f'/app/frames/frame_{i:04d}.png', frame)
        
    # 4. Encode video with ffmpeg
    os.makedirs('/app/output', exist_ok=True)
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-framerate', '16',
        '-i', '/app/frames/frame_%04d.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '/app/output/video.mp4'
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    
if __name__ == '__main__':
    solve()
