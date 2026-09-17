import cv2
import numpy as np
import imageio
import os

def solve():
    # Read the first frame
    img = cv2.imread('/app/first_frame.png')
    original_img = img.copy()
    
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Identify hollow circles
    hollow_circles = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        # Hollow circles have a child contour (hierarchy[0][i][2] != -1)
        # AND parent is -1 (top-level) to avoid counting the inner circle as a separate hollow circle
        # Let's be precise: Top level contours with a child.
        if area > 100 and hierarchy[0][i][3] == -1 and hierarchy[0][i][2] != -1:
            (x, y), r = cv2.minEnclosingCircle(cnt)
            hollow_circles.append((int(x), int(y), int(r)))

    # Sort hollow circles by y-coordinate for a nice top-to-bottom step-by-step
    hollow_circles.sort(key=lambda c: c[1])

    frames = []
    num_frames = 80
    fps = 16
    
    # We have `len(hollow_circles)` circles.
    # Let's dedicate some frames for each circle's animation, and some hold at the end.
    n_circles = len(hollow_circles)
    frames_per_circle = 16
    start_delay = 8
    
    ring_radius_offset = 20
    thickness = 5
    color = (255, 0, 0) # Red in RGB for imageio

    # We will generate frames in RGB
    base_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    for frame_idx in range(num_frames):
        # Create a copy of the base image for this frame
        current_frame = base_rgb.copy()
        
        # Determine how many circles are fully drawn, and if one is currently being drawn
        for i, (cx, cy, r) in enumerate(hollow_circles):
            circle_start_frame = start_delay + i * frames_per_circle
            circle_end_frame = circle_start_frame + frames_per_circle
            
            if frame_idx >= circle_end_frame:
                # Fully drawn
                cv2.circle(current_frame, (cx, cy), r + ring_radius_offset, color, thickness, cv2.LINE_AA)
            elif frame_idx >= circle_start_frame:
                # Being drawn
                progress = (frame_idx - circle_start_frame + 1) / frames_per_circle
                angle = int(360 * progress)
                cv2.ellipse(current_frame, (cx, cy), (r + ring_radius_offset, r + ring_radius_offset), 270, 0, angle, color, thickness, cv2.LINE_AA)
                
        frames.append(current_frame)
        
    # Make sure output directory exists
    os.makedirs('/app/output', exist_ok=True)
    
    # Write video
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, macro_block_size=None, codec='libx264', pixelformat='yuv420p')
    for f in frames:
        writer.append_data(f)
    writer.close()

if __name__ == '__main__':
    solve()
