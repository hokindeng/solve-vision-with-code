import cv2
import numpy as np
import os
import subprocess

def create_video():
    img_orig = cv2.imread('/app/first_frame.png')
    height, width, _ = img_orig.shape
    
    fps = 16
    total_frames = 60
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_out = '/app/temp_video.mp4'
    out = cv2.VideoWriter(temp_out, fourcc, fps, (width, height))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thick = 2
    
    seq_centers = [241, 421, 602, 783]
    seq_boxes = [
        (209, 305, 274, 370), # Seq 0
        (357, 273, 486, 402), # Seq 1
        (570, 305, 635, 370), # Seq 2
        (707, 262, 858, 413), # Seq 3
    ]
    texts = ["SMALL", "LARGE", "SMALL", "LARGE ?"]
    text_colors = [(0, 150, 0), (0, 150, 0), (0, 150, 0), (0, 0, 255)] # Last one red
    box_colors = [(0, 200, 0), (0, 200, 0), (0, 200, 0), (0, 0, 255)] # Last one red
    
    for frame_idx in range(total_frames):
        img = img_orig.copy()
        
        num_items = 0
        if frame_idx >= 8: num_items = 1
        if frame_idx >= 16: num_items = 2
        if frame_idx >= 24: num_items = 3
        if frame_idx >= 32: num_items = 4
        
        for i in range(num_items):
            # draw box
            x1, y1, x2, y2 = seq_boxes[i]
            cv2.rectangle(img, (x1, y1), (x2, y2), box_colors[i], 3)
            
            # draw text
            txt = texts[i]
            tsize, _ = cv2.getTextSize(txt, font, scale, thick)
            tx = seq_centers[i] - tsize[0] // 2
            ty = 450
            cv2.putText(img, txt, (tx, ty), font, scale, text_colors[i], thick)
            
        # Draw red circle
        if frame_idx >= 40:
            progress = min(1.0, (frame_idx - 40) / 10.0)
            angle = int(360 * progress)
            if angle > 0:
                cv2.ellipse(img, (626, 854), (115, 115), 0, 0, angle, (0, 0, 255), 4)
                
        out.write(img)
        
    out.release()
    
    os.makedirs('/app/output', exist_ok=True)
    subprocess.run([
        'ffmpeg', '-y', '-i', temp_out, 
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-crf', '18', '-preset', 'slower',
        '/app/output/video.mp4'
    ], check=True)

if __name__ == "__main__":
    create_video()
