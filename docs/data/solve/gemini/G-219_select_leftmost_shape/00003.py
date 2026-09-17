import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    
    # Detect the leftmost shape
    mask = cv2.inRange(img, np.array([0, 0, 0]), np.array([254, 254, 254]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shapes = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 10:
            x, y, w, h = cv2.boundingRect(cnt)
            shapes.append((x, y, w, h))
            
    # Sort by x coordinate to find the leftmost
    shapes.sort(key=lambda s: s[0])
    leftmost = shapes[0]
    
    x, y, w, h = leftmost
    center = (x + w // 2, y + h // 2)
    
    # Radius to enclose the bounding box with some padding
    radius = int(np.sqrt(w**2 + h**2) / 2) + 15
    
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', macro_block_size=None, pixelformat='yuv420p')
    
    num_frames = 48
    
    for i in range(num_frames):
        frame = img.copy()
        
        # Calculate angle for animation (from 0 to 360)
        angle = int(360 * i / (num_frames - 1))
        
        if angle > 0:
            # Draw ellipse progressively
            # We use angle=-90 to start from the top, and draw clockwise
            cv2.ellipse(frame, center, (radius, radius), -90, 0, angle, (0, 0, 255), 5, lineType=cv2.LINE_AA)
        
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == "__main__":
    solve()
