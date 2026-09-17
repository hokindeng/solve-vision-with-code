import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # The background is white (255), shapes are darker.
    # Inverse binary threshold makes shapes white, background black.
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shapes = []
    for c in contours:
        M = cv2.moments(c)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            area = cv2.contourArea(c)
            _, radius = cv2.minEnclosingCircle(c)
            shapes.append({'cx': cx, 'cy': cy, 'area': area, 'radius': radius})

    # Sort shapes by size from largest to smallest
    shapes.sort(key=lambda x: x['area'], reverse=True)
    
    # Target is the second largest shape (index 1)
    target = shapes[1]
    
    cx, cy = target['cx'], target['cy']
    r = int(target['radius'] + 15)
    
    frames = []
    num_frames = 40
    for i in range(num_frames):
        frame = img.copy()
        
        # Calculate the angle for the sweeping circle
        # i=0 -> 0 degrees (first frame unchanged)
        # i=39 -> 360 degrees (last frame fully drawn)
        end_angle = int(i / (num_frames - 1) * 360)
        
        if end_angle > 0:
            # Draw an expanding arc to simulate drawing the circle over time
            cv2.ellipse(frame, (cx, cy), (r, r), -90, 0, end_angle, (0, 0, 255), 10)
        
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == '__main__':
    solve()
