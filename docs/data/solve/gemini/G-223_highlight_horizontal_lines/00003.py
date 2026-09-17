import cv2
import numpy as np
import os
import subprocess

def solve():
    img_orig = cv2.imread('/app/first_frame.png')
    h_img, w_img = img_orig.shape[:2]
    
    # find lines
    gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    horiz = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > h:  # horizontal line
            horiz.append((x, y, w, h))
            
    os.makedirs('/app/output', exist_ok=True)
    out_file = '/app/output/video.mp4'
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    tmp_file = '/app/tmp.mp4'
    out = cv2.VideoWriter(tmp_file, fourcc, 16.0, (w_img, h_img))
    
    num_frames = 48
    start_frame = 4
    end_frame = 42
    
    for i in range(num_frames):
        frame = img_orig.copy()
        
        if i >= start_frame:
            if i > end_frame:
                progress = 1.0
            else:
                progress = (i - start_frame) / (end_frame - start_frame)
                
            angle_drawn = int(progress * 360)
            
            for x, y, w, h in horiz:
                center = (x + w//2, y + h//2)
                radius = int(max(w, h)/2 + 12)
                
                if angle_drawn > 0:
                    # Draw arc from top (-90 degrees) clockwise
                    cv2.ellipse(frame, center, (radius, radius), 0, -90, -90 + angle_drawn, (0, 0, 0), 6, cv2.LINE_AA)
                    
        out.write(frame)
        
    out.release()
    
    # Encode with ffmpeg to h264 yuv420p
    subprocess.run(['ffmpeg', '-y', '-i', tmp_file, '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', out_file], check=True)
    if os.path.exists(tmp_file):
        os.remove(tmp_file)

if __name__ == '__main__':
    solve()
