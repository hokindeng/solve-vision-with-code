import cv2
import numpy as np
import imageio
import math

def get_tangent_point(img):
    # Find background color (most common color)
    colors = img.reshape(-1, 3)
    # Get distinct colors
    unique_colors, counts = np.unique(colors, axis=0, return_counts=True)
    bg_color = unique_colors[np.argmax(counts)]
    
    circles = []
    
    for c in unique_colors:
        if np.array_equal(c, bg_color):
            continue
            
        mask = np.all(img == c, axis=-1)
        y, x = np.where(mask)
        if len(x) > 0:
            min_x, max_x = np.min(x), np.max(x)
            min_y, max_y = np.min(y), np.max(y)
            cx = (min_x + max_x) / 2.0
            cy = (min_y + max_y) / 2.0
            rx = (max_x - min_x) / 2.0
            
            # only consider reasonably sized circles
            if rx > 10:
                circles.append({'center': (cx, cy), 'radius': rx})
                
    # Find the two touching circles
    min_diff = float('inf')
    touching_pair = None
    
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            c1, c2 = circles[i], circles[j]
            dist = math.hypot(c1['center'][0] - c2['center'][0], c1['center'][1] - c2['center'][1])
            sum_r = c1['radius'] + c2['radius']
            diff = abs(dist - sum_r)
            if diff < min_diff:
                min_diff = diff
                touching_pair = (c1, c2)
                
    if touching_pair:
        c1, c2 = touching_pair
        r1 = c1['radius']
        r2 = c2['radius']
        tx = c1['center'][0] + (c2['center'][0] - c1['center'][0]) * r1 / (r1 + r2)
        ty = c1['center'][1] + (c2['center'][1] - c1['center'][1]) * r1 / (r1 + r2)
        return (int(round(tx)), int(round(ty)))
    
    return (537, 298) # fallback

def make_video():
    img = cv2.imread('/app/first_frame.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    center = get_tangent_point(img)
    
    fps = 16
    frames_count = 60
    
    radius = 30
    thickness = 4
    color = (0, 0, 0) 
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    for i in range(frames_count):
        frame = img_rgb.copy()
        if i > 0:
            angle = i / (frames_count - 1) * 360.0
            cv2.ellipse(frame, center, (radius, radius), -90, 0, angle, color, thickness, cv2.LINE_AA)
            
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    make_video()
