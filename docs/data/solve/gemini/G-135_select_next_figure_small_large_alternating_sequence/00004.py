import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    top_shapes = []
    bottom_shapes = []
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 20 < w < 150 and 20 < h < 150:
            if y < 512:
                top_shapes.append((x, y, w, h, c))
            else:
                bottom_shapes.append((x, y, w, h, c))
                
    # Filter out nested contours (if any)
    def filter_nested(shapes):
        res = []
        for s1 in shapes:
            x1, y1, w1, h1, c1 = s1
            is_nested = False
            for s2 in shapes:
                x2, y2, w2, h2, c2 = s2
                if s1 == s2:
                    continue
                # if s1 is inside s2
                if x1 >= x2 and y1 >= y2 and x1+w1 <= x2+w2 and y1+h1 <= y2+h2:
                    # check if they are effectively the same bounding box
                    if not (x1 == x2 and y1 == y2 and w1 == w2 and h1 == h2):
                        is_nested = True
            if not is_nested:
                res.append(s1)
        return res
        
    top_shapes = filter_nested(top_shapes)
    bottom_shapes = filter_nested(bottom_shapes)
    
    if top_shapes:
        top_colors = []
        for x, y, w, h, c in top_shapes:
            cx, cy = x + w//2, y + h//2
            color = img[cy, cx].tolist()
            top_colors.append(color)
            
        target_color = top_colors[0]
        
        max_area = 0
        for x, y, w, h, c in top_shapes:
            area = cv2.contourArea(c)
            if area > max_area:
                max_area = area
                
        best_match = None
        min_diff = float('inf')
        
        for x, y, w, h, c in bottom_shapes:
            cx, cy = x + w//2, y + h//2
            color = img[cy, cx].tolist()
            
            color_diff = sum([abs(color[i] - target_color[i]) for i in range(3)])
            if color_diff > 30:
                continue
                
            area = cv2.contourArea(c)
            area_diff = abs(area - max_area)
            
            if area_diff < min_diff:
                min_diff = area_diff
                best_match = (x, y, w, h)
                
        if best_match:
            x, y, w, h = best_match
            # We found the shape, wait, we actually need the center of the option box!
            # The option box contains the shape. We can just use the shape's center,
            # which is concentric with the option box in this task.
            center = (x + w//2, y + h//2)
        else:
            center = (856, 854)
    else:
        center = (856, 854)
        
    radius = 100
    color = (0, 0, 255) # Red
    thickness = 6
    
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    
    for i in range(60):
        frame = img.copy()
        if i > 0:
            end_angle = -90 + (i / 59.0) * 360
            cv2.ellipse(frame, center, (radius, radius), 0, -90, end_angle, color, thickness, cv2.LINE_AA)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    solve()
