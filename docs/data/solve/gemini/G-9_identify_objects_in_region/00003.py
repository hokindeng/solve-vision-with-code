import cv2
import numpy as np
import imageio
import os

def get_angle(pt, cx, cy):
    x, y = pt
    # arctan2 is from -pi to pi. y is down, x is right.
    a = np.arctan2(y - cy, x - cx)
    # Adjust so 0 is at top (-pi/2) and goes clockwise
    a_adjusted = (a - (-np.pi/2)) % (2 * np.pi)
    return a_adjusted

def solve():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        print("Error: Could not read /app/first_frame.png")
        return
        
    # 1. Find the circular region (black outline)
    black_mask = cv2.inRange(img, np.array([0,0,0]), np.array([0,0,0]))
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    circ_region_cnt = None
    circ_region_center = None
    circ_region_radius = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area > 10000 and perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.85: # It's a circle
                (x,y), r = cv2.minEnclosingCircle(cnt)
                circ_region_center = (x,y)
                circ_region_radius = r
                circ_region_cnt = cnt
                break
                
    if circ_region_center is None:
        print("Could not find circular region!")
        return

    # 2. Find all gray borders (objects)
    gray_mask = cv2.inRange(img, np.array([80,80,80]), np.array([80,80,80]))
    gray_contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    target_contours = []
    
    for cnt in gray_contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area > 10 and perimeter > 0:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
            else:
                (cx, cy), _ = cv2.minEnclosingCircle(cnt)
            
            # Check if inside circular region
            dist = np.hypot(cx - circ_region_center[0], cy - circ_region_center[1])
            if dist < circ_region_radius:
                # Check if it's a circle
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.85:
                    target_contours.append((cnt, cx, cy))

    # 3. Create frames
    os.makedirs('/app/output', exist_ok=True)
    num_frames = 40
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    
    for i in range(num_frames):
        frame = img.copy()
        progress = i / (num_frames - 1) if num_frames > 1 else 1.0
        current_max_angle = progress * 2 * np.pi
        
        if progress > 0:
            for cnt, cx, cy in target_contours:
                # Mask for just this object's gray pixels
                obj_mask = np.zeros_like(gray_mask)
                cv2.drawContours(obj_mask, [cnt], -1, 255, -1)
                
                # We want to strictly recolor only the gray pixels to preserve shape
                obj_gray_pixels = cv2.bitwise_and(gray_mask, obj_mask)
                
                ys, xs = np.where(obj_gray_pixels > 0)
                for y, x in zip(ys, xs):
                    if progress == 1.0:
                        frame[y, x] = [0, 255, 0]
                    else:
                        angle = get_angle((x, y), cx, cy)
                        if angle <= current_max_angle:
                            frame[y, x] = [0, 255, 0] # Green in BGR
                        
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()
    print("Video generated successfully.")

if __name__ == '__main__':
    solve()
