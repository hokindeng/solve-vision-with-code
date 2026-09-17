import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    approx = cv2.approxPolyDP(contours[0], 0.01 * cv2.arcLength(contours[0], True), True)
    pts = approx.reshape(-1, 2)
    
    # Calculate lengths of edges
    n = len(pts)
    edges = []
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i+1)%n]
        dist = np.linalg.norm(p1 - p2)
        edges.append({
            'dist': dist,
            'p1': p1,
            'p2': p2
        })
        
    # Sort edges by length (shortest first)
    edges.sort(key=lambda x: x['dist'])
    
    frames = []
    
    # Frame 0
    frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Frames 1-20
    for edge_idx, edge in enumerate(edges):
        p1 = edge['p1'].astype(np.float32)
        p2 = edge['p2'].astype(np.float32)
        
        for step in range(1, 5):
            frame = img.copy()
            pt = p1 + (p2 - p1) * (step / 4.0)
            pt = tuple(np.round(pt).astype(int))
            start_pt = tuple(np.round(p1).astype(int))
            
            cv2.line(frame, start_pt, pt, (0, 255, 255), 5)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
    # Frames 21-24
    longest_edge = edges[-1]
    p1 = longest_edge['p1']
    p2 = longest_edge['p2']
    midpoint = tuple(np.round((p1 + p2) / 2.0).astype(int))
    
    alphas = [0.75, 0.5, 0.25, 0.0]
    radii = [4, 8, 12, 12]
    
    for alpha, r in zip(alphas, radii):
        frame = img.copy()
        
        if alpha > 0:
            overlay = frame.copy()
            cv2.line(overlay, tuple(np.round(p1).astype(int)), tuple(np.round(p2).astype(int)), (0, 255, 255), 5)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
        cv2.circle(frame, midpoint, r, (0, 0, 255), -1)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    for f in frames:
        writer.append_data(f)
    writer.close()
    
    print(f"Generated {len(frames)} frames")

if __name__ == '__main__':
    solve()
