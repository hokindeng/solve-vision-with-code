import cv2
import numpy as np
import os
import shutil

def main():
    os.makedirs('/app/output', exist_ok=True)
    tmp_dir = '/app/tmp_frames'
    os.makedirs(tmp_dir, exist_ok=True)
    
    img_path = '/app/first_frame.png'
    first_frame = cv2.imread(img_path)
    
    # Detect horizontal lines
    mask = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY) < 250
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    horizontal_lines = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > h and w > 20:
            cx, cy = x + w//2, y + h//2
            r = int(max(w, h)/2) + 20
            horizontal_lines.append((cx, cy, r))
            
    # Video setup
    fps = 16
    total_frames = 48
    draw_frames = 40
    
    for i in range(total_frames):
        frame = first_frame.copy()
        
        # calculate end angle
        if i >= draw_frames:
            sweep = 360
        else:
            sweep = int(360 * (i / draw_frames))
            
        for cx, cy, r in horizontal_lines:
            if sweep > 0:
                cv2.ellipse(frame, (cx, cy), (r, r), 0, 0, sweep, (0, 0, 0), 5, cv2.LINE_AA)
                
        cv2.imwrite(f"{tmp_dir}/frame_{i:04d}.png", frame)
        
    out_path = '/app/output/video.mp4'
    os.system(f"ffmpeg -y -framerate {fps} -i {tmp_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 17 {out_path} -loglevel quiet")
    
    shutil.rmtree(tmp_dir)

if __name__ == '__main__':
    main()
