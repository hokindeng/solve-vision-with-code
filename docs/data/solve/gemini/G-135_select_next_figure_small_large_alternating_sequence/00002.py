import cv2
import numpy as np
import imageio

def generate_video():
    img = cv2.imread('/app/first_frame.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    
    fps = 16
    total_frames = 60
    
    cx, cy = 626, 854
    radius = 65
    thickness = 6
    color = (255, 0, 0) # Red in RGB
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, codec='libx264', pixelformat='yuv420p')
    
    for i in range(total_frames):
        frame = img_rgb.copy()
        
        # We can use cv2.ellipse on the RGB frame
        angle = int((i / (total_frames - 1)) * 360)
        if angle > 0:
            cv2.ellipse(frame, (cx, cy), (radius, radius), 0, -90, -90 + angle, color, thickness, cv2.LINE_AA)
            
        writer.append_data(frame)
        
    writer.close()

if __name__ == "__main__":
    import os
    os.makedirs('/app/output', exist_ok=True)
    generate_video()
