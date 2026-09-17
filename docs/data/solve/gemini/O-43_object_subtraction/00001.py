import cv2
import numpy as np
import os
import imageio

def solve():
    img = cv2.imread('/app/first_frame.png')
    
    # Identify orange pixels
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    bg_color = np.array([255, 255, 255], dtype=np.float32)
    
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer(
        '/app/output/video.mp4', 
        fps=16, 
        codec='libx264',
        pixelformat='yuv420p',
        macro_block_size=None,
        ffmpeg_params=['-crf', '17']
    )
    
    frames = 96
    
    for i in range(frames):
        alpha = 1.0 - (i / (frames - 1))
        
        frame = img.copy()
        
        # Find orange pixels
        orange_pixels = mask > 0
        original_colors = frame[orange_pixels].astype(np.float32)
        
        # Blend with background
        blended = original_colors * alpha + bg_color * (1 - alpha)
        frame[orange_pixels] = np.clip(blended, 0, 255).astype(np.uint8)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    solve()
