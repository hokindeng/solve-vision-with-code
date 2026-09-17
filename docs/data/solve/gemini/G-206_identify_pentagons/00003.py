import cv2
import numpy as np
import os

def main():
    # Read first frame
    img = cv2.imread('/app/first_frame.png')
    
    # Detect the pentagon
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pentagon_contour = None
    pentagon_center = None
    pentagon_radius = 0
    
    for c in contours:
        area = cv2.contourArea(c)
        if area < 100: continue
        
        epsilon = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if len(approx) == 5:
            pentagon_contour = c
            (x, y), radius = cv2.minEnclosingCircle(c)
            pentagon_center = (int(x), int(y))
            pentagon_radius = int(radius) + 5
            break
            
    if pentagon_center is None:
        print("Pentagon not found")
        return
        
    os.makedirs('/app/output', exist_ok=True)
    
    # Generate frames
    frames = []
    num_frames = 30
    
    for i in range(num_frames):
        frame = img.copy()
        current_radius = int(pentagon_radius * (i / (num_frames - 1)))
        
        if current_radius > 0:
            # Draw red circle (BGR: 0, 0, 255)
            cv2.circle(frame, pentagon_center, current_radius, (0, 0, 255), 4, lineType=cv2.LINE_AA)
            
        frames.append(frame)
        
    # Write to video
    # Use imageio to write video
    import imageio
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', format='FFMPEG', macro_block_size=None, pixelformat='yuv420p')
    for frame in frames:
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    writer.close()

if __name__ == "__main__":
    main()
