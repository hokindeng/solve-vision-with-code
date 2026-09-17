import cv2
import numpy as np
import imageio
import os

def main():
    # 1. Read the first frame
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("Could not read /app/first_frame.png")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Find the circles
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circles = []
    for c in contours:
        (x, y), radius = cv2.minEnclosingCircle(c)
        area = cv2.contourArea(c)
        if area > 100:
            circles.append({
                'x': int(round(x)), 
                'y': int(round(y)), 
                'r': int(round(radius)), 
                'area': area
            })

    # Sort by area to find the second largest
    circles.sort(key=lambda c: c['area'], reverse=True)
    if len(circles) < 2:
        raise ValueError("Could not find at least 2 circles")
        
    target = circles[1]  # Second largest

    # 3. Create the video
    fps = 16
    num_frames = 40
    output_path = '/app/output/video.mp4'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=None)

    center = (target['x'], target['y'])
    radius = target['r'] + 15
    color = (255, 0, 0)  # Red in RGB
    thickness = 8

    for i in range(num_frames):
        frame = img_rgb.copy()
        
        # Calculate the end angle for the current frame
        end_angle = int(360 * i / (num_frames - 1))
        
        if end_angle > 0:
            cv2.ellipse(frame, center, (radius, radius), -90, 0, end_angle, color, thickness, cv2.LINE_AA)
            
        writer.append_data(frame)

    writer.close()

if __name__ == '__main__':
    main()
