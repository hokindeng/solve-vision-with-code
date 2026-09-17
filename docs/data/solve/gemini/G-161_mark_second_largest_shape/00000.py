import cv2
import numpy as np
import imageio
import os

def main():
    img_path = '/app/first_frame.png'
    output_path = '/app/output/video.mp4'
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read {img_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shapes = []
    for c in contours:
        # We can calculate the minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(c)
        area = cv2.contourArea(c)
        if radius > 10 and area > 100:
            shapes.append(((int(x), int(y)), int(radius), area))
            
    # Sort by area from largest to smallest
    shapes.sort(key=lambda s: s[2], reverse=True)
    
    if len(shapes) < 2:
        print("Could not find at least 2 shapes")
        return
        
    # The second largest
    target_shape = shapes[1]
    center = target_shape[0]
    radius = target_shape[1]
    
    # We want to circle it. We will use a radius slightly larger than the shape itself.
    draw_radius = radius + 20
    
    # Animation parameters
    num_frames = 40
    fps = 16
    
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p')
    
    for i in range(num_frames):
        frame = img.copy()
        
        # Calculate how much of the circle to draw
        # end_angle goes from 0 to 360 over the frames
        # For frame 0, we can draw 0 or a small amount
        # For frame num_frames-1, we draw 360
        progress = i / (num_frames - 1)
        end_angle = int(360 * progress)
        
        if end_angle > 0:
            # cv2.ellipse signature: (img, center, axes, angle, startAngle, endAngle, color, thickness, lineType)
            # Axes is (width/2, height/2), so we pass (draw_radius, draw_radius)
            # We want the circle to start from top (-90 degrees) and go clockwise
            cv2.ellipse(frame, center, (draw_radius, draw_radius), -90, 0, end_angle, (0, 0, 255), 8, cv2.LINE_AA)
            
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()
    print("Video generated at", output_path)

if __name__ == '__main__':
    main()
