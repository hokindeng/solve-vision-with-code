import cv2
import numpy as np
import math
import imageio
import os

def get_vertices(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Use the largest contour in case of noise
    cnt = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(cnt, 0.005 * cv2.arcLength(cnt, True), True)
    pts = [p[0] for p in approx]
    return pts, img

def generate_video():
    input_path = '/app/first_frame.png'
    output_path = '/app/output/video.mp4'
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    pts, base_img = get_vertices(input_path)
    
    # Calculate lengths and midpoints
    edges = []
    # Calculate center of polygon for normal direction
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    
    for i in range(len(pts)):
        p1 = tuple(int(x) for x in pts[i])
        p2 = tuple(int(x) for x in pts[(i+1)%len(pts)])
        d = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
        mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
        
        # Normal for text
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.hypot(dx, dy)
        if length > 0:
            nx, ny = -dy/length, dx/length
        else:
            nx, ny = 0, 0
            
        # Ensure normal points outward
        if (mx + nx - cx)**2 + (my + ny - cy)**2 < (mx - nx - cx)**2 + (my - ny - cy)**2:
            nx, ny = -nx, -ny
            
        edges.append({
            'p1': p1, 'p2': p2, 'len': d, 'mid': (int(mx), int(my)), 'norm': (nx, ny)
        })
        
    longest_idx = max(range(len(edges)), key=lambda i: edges[i]['len'])
    
    frames = []
    
    # Frames 0-2: initial (3 frames)
    for _ in range(3):
        frames.append(base_img.copy())
        
    highlight_color = (0, 255, 255) # Yellow in BGR
    highlight_thickness = 8
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    text_thickness = 2
    
    for i in range(len(edges)):
        edge = edges[i]
        p1 = edge['p1']
        p2 = edge['p2']
        pmid = edge['mid']
        nx, ny = edge['norm']
        length_str = f"{int(round(edge['len']))}"
        
        # Calculate text position
        offset = 40
        tx = int(pmid[0] + nx * offset)
        ty = int(pmid[1] + ny * offset)
        text_size = cv2.getTextSize(length_str, font, font_scale, text_thickness)[0]
        tx = tx - text_size[0] // 2
        ty = ty + text_size[1] // 2
        
        # Frame 1: half measured
        frame1 = base_img.copy()
        cv2.line(frame1, p1, pmid, highlight_color, highlight_thickness)
        cv2.putText(frame1, length_str, (tx, ty), font, font_scale, (0, 0, 0), text_thickness, cv2.LINE_AA)
        frames.append(frame1)
        
        # Frame 2: fully measured
        frame2 = base_img.copy()
        cv2.line(frame2, p1, p2, highlight_color, highlight_thickness)
        cv2.putText(frame2, length_str, (tx, ty), font, font_scale, (0, 0, 0), text_thickness, cv2.LINE_AA)
        frames.append(frame2)
        
    # Frame 19-24: final result (red circle at longest midpoint)
    final_frame = base_img.copy()
    longest_edge = edges[longest_idx]
    lmx, lmy = longest_edge['mid']
    cv2.circle(final_frame, (lmx, lmy), 12, (0, 0, 255), -1) # Red in BGR
    
    while len(frames) < 25:
        frames.append(final_frame.copy())
        
    # print(f"Total frames: {len(frames)}")
    
    # Write frames
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    imageio.mimwrite(output_path, frames_rgb, fps=16, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    generate_video()
