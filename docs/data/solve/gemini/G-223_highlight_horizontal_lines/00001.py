import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    
    pixels = img.reshape(-1, 3)
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    bg_color = colors[np.argmax(counts)]
    
    mask = np.any(img != bg_color, axis=-1).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    horizontal_lines = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > h:
            horizontal_lines.append((x, y, w, h))
            
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    
    num_frames = 48
    
    for i in range(num_frames):
        frame = img.copy()
        
        angle = int(360 * i / (num_frames - 1))
        
        if angle > 0:
            for (x, y, w, h) in horizontal_lines:
                center = (x + w // 2, y + h // 2)
                radius = int(np.sqrt((w/2)**2 + (h/2)**2)) + 15
                
                cv2.ellipse(frame, center, (radius, radius), -90, 0, angle, (0, 0, 0), 4, cv2.LINE_AA)
                
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    solve()
