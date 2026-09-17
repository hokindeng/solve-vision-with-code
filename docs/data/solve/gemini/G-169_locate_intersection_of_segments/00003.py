import cv2
import numpy as np
import os
import subprocess
import tempfile

def find_intersection(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist)
    return max_loc

def create_video():
    image_path = '/app/first_frame.png'
    output_dir = '/app/output'
    os.makedirs(output_dir, exist_ok=True)
    
    img = cv2.imread(image_path)
    center = find_intersection(image_path)
    
    fps = 16
    total_frames = 30
    
    with tempfile.TemporaryDirectory() as frames_dir:
        for i in range(total_frames):
            frame = img.copy()
            
            start_draw_frame = 2
            end_draw_frame = 25
            
            if i >= start_draw_frame:
                if i >= end_draw_frame:
                    end_angle = 360
                else:
                    progress = (i - start_draw_frame) / (end_draw_frame - start_draw_frame)
                    end_angle = int(360 * progress)
                    
                if end_angle > 0:
                    cv2.ellipse(frame, center, (30, 30), 0, 0, end_angle, (0, 0, 255), 4, cv2.LINE_AA)
            
            frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
            cv2.imwrite(frame_path, frame)
            
        output_video = os.path.join(output_dir, 'video.mp4')
        cmd = [
            'ffmpeg', '-y', '-framerate', str(fps), '-i', os.path.join(frames_dir, 'frame_%04d.png'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', output_video
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == '__main__':
    create_video()
