import cv2
import numpy as np
import imageio
import os

def solve():
    # Load first frame
    img_path = '/app/first_frame.png'
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image at {img_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Identify the asymmetrical shape
    max_asym = -1
    asym_contour = None
    for c in contours:
        m = cv2.moments(c)
        if m['m00'] == 0:
            continue
        hu = cv2.HuMoments(m).flatten()
        # Hu[2] to Hu[6] are robust indicators of asymmetry
        score = sum(abs(h) for h in hu[2:7])
        if score > max_asym:
            max_asym = score
            asym_contour = c
            
    if asym_contour is None:
        raise ValueError("No contours found.")
        
    # Get the bounding circle for the asymmetrical shape
    (x, y), radius = cv2.minEnclosingCircle(asym_contour)
    center = (int(x), int(y))
    radius = int(radius) + 15  # Add padding for visual clearance
    
    # Prepare video writer
    os.makedirs('/app/output', exist_ok=True)
    out_path = '/app/output/video.mp4'
    
    # imageio writer with required codec and pixel format
    writer = imageio.get_writer(out_path, fps=16, codec='libx264', pixelformat='yuv420p')
    
    num_frames = 16
    for i in range(num_frames):
        # Create a copy of the original frame
        frame = img.copy()
        
        # Calculate the angle for the current frame (0 to 360)
        angle = int(360 * i / (num_frames - 1))
        
        # Draw the animated red circle
        if angle > 0:
            cv2.ellipse(frame, center, (radius, radius), 0, 0, angle, (0, 0, 255), 6, cv2.LINE_AA)
            
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    solve()
