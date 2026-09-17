import cv2
import numpy as np
import imageio
import os

def get_horizontal_lines(img):
    # Find background color assuming top-left is background
    bg_color = img[0, 0]
    
    # Calculate difference from background to find lines
    diff = np.abs(img.astype(np.int32) - bg_color.astype(np.int32)).sum(axis=2)
    thresh = (diff > 0).astype(np.uint8) * 255
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    
    horizontal_lines = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        # A horizontal line will have width > height
        if w > h:
            cx, cy = int(x + w / 2), int(y + h / 2)
            # Make the radius slightly larger than half the width
            radius = int(max(w, h) / 2) + 20
            horizontal_lines.append((cx, cy, radius))
            
    return horizontal_lines

def main():
    os.makedirs('/app/output', exist_ok=True)
    img = cv2.imread('/app/first_frame.png')
    
    if img is None:
        raise FileNotFoundError("Could not read /app/first_frame.png")
        
    lines = get_horizontal_lines(img)
    
    fps = 16
    duration = 3.0
    num_frames = int(fps * duration)
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, codec='libx264', pixelformat='yuv420p')
    
    for i in range(num_frames):
        frame = img.copy()
        
        # Pacing: from 0 to 360 degrees
        progress = i / (num_frames - 1)
        end_angle = int(360 * progress)
        
        for cx, cy, radius in lines:
            if end_angle > 0:
                cv2.ellipse(frame, (cx, cy), (radius, radius), 0, 0, end_angle, (0, 0, 0), 4, cv2.LINE_AA)
                
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    main()
