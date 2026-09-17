import cv2
import numpy as np
import imageio
import os

def solve():
    first_frame_path = '/app/first_frame.png'
    output_path = '/app/output/video.mp4'
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    image = cv2.imread(first_frame_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Identify background color from top-left pixel
    bg_color = image[0, 0]
    
    # Isolate shapes
    diff = cv2.absdiff(image, np.full_like(image, bg_color))
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate sizes and get centers
    circles = []
    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        circles.append(( (int(x), int(y)), radius ))
        
    # Sort circles by radius in descending order
    circles.sort(key=lambda c: c[1], reverse=True)
    
    # Second largest is at index 1
    second_largest = circles[1]
    center = second_largest[0]
    radius = int(second_largest[1]) + 15
    
    num_frames = 40
    fps = 16
    
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p')
    
    for i in range(num_frames):
        frame = image_rgb.copy()
        
        # Calculate end angle for the current frame
        end_angle = int(360 * (i / (num_frames - 1)))
        
        if end_angle > 0:
            # Draw the arc in red (RGB: 255, 0, 0)
            cv2.ellipse(frame, center, (radius, radius), 0, 0, end_angle, (255, 0, 0), 8, cv2.LINE_AA)
            
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    solve()
