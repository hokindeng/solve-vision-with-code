import cv2
import numpy as np
import os
import subprocess
import shutil

def find_unique_shape(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(c) for c in contours]
    
    from collections import defaultdict
    area_counts = defaultdict(list)
    for i, a in enumerate(areas):
        matched = False
        for k in list(area_counts.keys()):
            if abs(a - k) < 500:
                area_counts[k].append(i)
                matched = True
                break
        if not matched:
            area_counts[a].append(i)

    unique_idx = -1
    for k, v in area_counts.items():
        if len(v) == 1:
            unique_idx = v[0]
            break
            
    return contours[unique_idx]

def make_video():
    first_frame_path = '/app/first_frame.png'
    output_path = '/app/output/video.mp4'
    frames_dir = '/tmp/frames'
    
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    
    img = cv2.imread(first_frame_path)
    
    unique_cnt = find_unique_shape(img)
    x, y, bw, bh = cv2.boundingRect(unique_cnt)
    
    cx = x + bw // 2
    cy = y + bh // 2
    radius = int(max(bw, bh) / 2) + 20
    
    fps = 16
    total_frames = 60
    
    start_draw_frame = 5
    end_draw_frame = 55
    
    for i in range(total_frames):
        frame = img.copy()
        
        if i >= start_draw_frame:
            progress = (i - start_draw_frame) / (end_draw_frame - start_draw_frame)
            progress = min(max(progress, 0.0), 1.0)
            end_angle = int(progress * 360)
            
            if end_angle > 0:
                cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, end_angle, (0, 0, 255), 10, cv2.LINE_AA)
                
        cv2.imwrite(f"{frames_dir}/frame_{i:04d}.png", frame)
        
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps), "-i", f"{frames_dir}/frame_%04d.png",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    make_video()
