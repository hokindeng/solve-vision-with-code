import cv2
import numpy as np
import imageio
import math
import os

def draw_partial_polygon(img, pts, color, thickness, progress):
    if progress <= 0:
        return
    pts = list(pts)
    pts.append(pts[0])
    
    lengths = []
    for i in range(len(pts)-1):
        x1, y1 = pts[i]
        x2, y2 = pts[i+1]
        lengths.append(math.hypot(x2-x1, y2-y1))
        
    total_length = sum(lengths)
    target_length = total_length * progress
    
    current_length = 0
    for i in range(len(pts)-1):
        x1, y1 = pts[i]
        x2, y2 = pts[i+1]
        L = lengths[i]
        
        if target_length >= current_length + L - 1e-6:
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            current_length += L
        else:
            rem = target_length - current_length
            ratio = rem / L if L > 0 else 0
            px = x1 + (x2 - x1) * ratio
            py = y1 + (y2 - y1) * ratio
            cv2.line(img, (int(x1), int(y1)), (int(px), int(py)), color, thickness)
            break

def main():
    img_bgr = cv2.imread('/app/first_frame.png')
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    square_regions = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 200 and h > 200:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            if len(approx) == 4 and 0.8 <= w/h <= 1.2:
                square_regions.append(c)
                
    if not square_regions:
        print("No square region found!")
        return
        
    square_region_c = max(square_regions, key=cv2.contourArea)
    
    target_squares = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 10 < w < 150 and 10 < h < 150:
            cx, cy = x + w/2.0, y + h/2.0
            if cv2.pointPolygonTest(square_region_c, (cx, cy), False) > 0:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.04 * peri, True)
                if len(approx) == 4 and 0.8 <= w/h <= 1.2:
                    pts = [(pt[0][0], pt[0][1]) for pt in approx]
                    target_squares.append(pts)
                    
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    num_frames = 40
    for i in range(num_frames):
        frame = img_rgb.copy()
        progress = i / float(num_frames - 1)
        for pts in target_squares:
            draw_partial_polygon(frame, pts, (0, 255, 0), 4, progress)
        writer.append_data(frame)
        
    writer.close()
    print("Done")

if __name__ == '__main__':
    main()
