import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("Could not read /app/first_frame.png")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Determine background color to handle light or dark backgrounds
    bg_color = gray[0, 0]
    if bg_color < 128:
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    else:
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pentagon_cnt = None
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        area = cv2.contourArea(cnt)
        if area > 100 and len(approx) == 5:
            pentagon_cnt = cnt
            break

    if pentagon_cnt is None:
        print("No pentagon found!")
        return

    # Find the enclosing circle for the pentagon
    (x, y), radius = cv2.minEnclosingCircle(pentagon_cnt)
    center = (int(x), int(y))
    # Encircle completely by adding a small padding to the radius
    target_radius = int(radius) + 15
    
    os.makedirs('/app/output', exist_ok=True)
    out_path = '/app/output/video.mp4'
    
    # Create the video writer
    writer = imageio.get_writer(out_path, fps=16, codec='libx264', pixelformat='yuv420p')
    
    frames_count = 30
    for i in range(frames_count):
        frame = img.copy()
        
        # Calculate current radius
        current_radius = int(target_radius * (i / (frames_count - 1)))
        
        if current_radius > 0:
            # Draw the red circle (OpenCV uses BGR)
            cv2.circle(frame, center, current_radius, (0, 0, 255), 4)
            
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    solve()
