import cv2
import numpy as np
import imageio
import os

def get_hollow_circles(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    hollow = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > 10:
            if hierarchy[0][i][3] == -1 and hierarchy[0][i][2] != -1:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                hollow.append((int(x), int(y), int(radius)))
                
    return img, hollow

def main():
    img, hollow_circles = get_hollow_circles('/app/first_frame.png')
    # Sort by y, then x (top to bottom reading order)
    hollow_circles.sort(key=lambda c: (c[1], c[0]))

    num_circles = len(hollow_circles)
    total_frames = 80
    
    frames = []
    ring_radius = 85
    ring_thickness = 5
    red_color = (255, 0, 0) # RGB
    
    state_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Ensure first frame is exactly the untouched original image
    frames.append(state_img.copy())
    
    # We have 79 frames left to distribute
    # Allocate 15 frames per circle for the drawing animation
    frames_per_circle = 15
    
    for i in range(num_circles):
        x, y, r = hollow_circles[i]
        for f in range(frames_per_circle):
            frame = state_img.copy()
            
            # Calculate angle for current frame (0 to 360)
            angle = int(360 * (f + 1) / frames_per_circle)
            
            # Draw the arc
            cv2.ellipse(frame, (x, y), (ring_radius, ring_radius), -90, 0, angle, red_color, ring_thickness)
            
            frames.append(frame)
            
            # Update state_img when the circle is complete
            if f == frames_per_circle - 1:
                state_img = frame.copy()
                
    # Pad with the final state to reach exactly 80 frames
    while len(frames) < total_frames:
        frames.append(state_img.copy())

    # Save to video
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == '__main__':
    main()
