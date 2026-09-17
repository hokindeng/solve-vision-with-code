import cv2
import numpy as np
import imageio
import os

def main():
    input_path = '/app/first_frame.png'
    output_path = '/app/output/video.mp4'
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not read {input_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No shapes found in the image")
        
    # Find the leftmost contour based on bounding box
    leftmost_c = min(contours, key=lambda c: cv2.boundingRect(c)[0])
    
    x, y, w, h = cv2.boundingRect(leftmost_c)
    cx, cy = x + w // 2, y + h // 2
    # Determine circle radius, slightly larger than the shape
    r = max(w, h) // 2 + 20
    
    num_frames = 48
    fps = 16
    
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', format='FFMPEG', macro_block_size=None, pixelformat='yuv420p')
    
    # Convert base image to RGB for imageio
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for i in range(num_frames):
        frame = img_rgb.copy()
        
        # Calculate angle for progressive drawing (0 to 360)
        end_angle = (i / (num_frames - 1)) * 360
        
        if end_angle > 0:
            # OpenCV ellipse: center, axes, angle, startAngle, endAngle, color, thickness, lineType
            # Red color in RGB is (255, 0, 0)
            cv2.ellipse(frame, (cx, cy), (r, r), -90, 0, end_angle, (255, 0, 0), 8, cv2.LINE_AA)
            
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    main()
