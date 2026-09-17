import cv2
import numpy as np
import imageio
import os

def main():
    # 1. Read the first frame
    first_frame_path = '/app/first_frame.png'
    img = cv2.imread(first_frame_path)
    
    # 2. Identify hollow points
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    contours, hierarchy = cv2.findContours(inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    hollow_pts = []
    for i, c in enumerate(contours):
        h = hierarchy[0][i]
        # h[3] == -1 means it is a root contour in the inverted image (an outer boundary)
        if h[3] == -1:
            has_child = h[2] != -1
            if has_child:
                (x, y), radius = cv2.minEnclosingCircle(c)
                hollow_pts.append((int(x), int(y), int(radius)))
                
    # Sort points left to right
    hollow_pts.sort(key=lambda p: p[0])
    
    # 3. Generate frames
    total_frames = 80
    fps = 16
    out_path = '/app/output/video.mp4'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    writer = imageio.get_writer(out_path, fps=fps, codec='libx264', pixelformat='yuv420p')
    
    for frame_idx in range(total_frames):
        frame_img = img.copy()
        
        # Determine which circles are fully drawn and which is being drawn
        if frame_idx < 8:
            pass # nothing drawn yet
        else:
            # For each point, calculate its progress
            for pt_idx, (x, y, r) in enumerate(hollow_pts):
                start_frame = 8 + pt_idx * 20
                end_frame = start_frame + 20
                
                if frame_idx >= end_frame:
                    # Fully drawn
                    cv2.circle(frame_img, (x, y), r + 15, (0, 0, 255), 6, cv2.LINE_AA)
                elif frame_idx >= start_frame:
                    # Partially drawn
                    progress = (frame_idx - start_frame + 1) / 20.0
                    end_angle = -90 + progress * 360
                    cv2.ellipse(frame_img, (x, y), (r + 15, r + 15), 0, -90, end_angle, (0, 0, 255), 6, cv2.LINE_AA)
                    
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    main()
