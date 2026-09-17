import cv2
import numpy as np
import imageio
import os

def get_normal(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    l = np.hypot(dx, dy)
    if l == 0: return np.array([0.0, 0.0])
    return np.array([-dy/l, dx/l])

def main():
    img_orig = cv2.imread('/app/first_frame.png')
    if img_orig is None:
        raise ValueError("Could not read /app/first_frame.png")
    
    # Automatically detect the polygon
    gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour
    cnt = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(cnt, 0.001 * cv2.arcLength(cnt, True), True)
    pts = approx.reshape(-1, 2).astype(float)
    n = len(pts)
    center = np.mean(pts, axis=0)
    
    edges = []
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i+1)%n]
        dist = np.linalg.norm(p1 - p2)
        edges.append({'dist': dist, 'p1': p1, 'p2': p2, 'idx': i})
        
    sorted_edges = sorted(edges, key=lambda x: x['dist'], reverse=True)
    longest_edge_idx = sorted_edges[0]['idx']
    
    frames = []
    
    for f in range(25):
        img = img_orig.copy()
        frames_per_edge = 3
        
        for i in range(n):
            start_f = i * frames_per_edge + 1
            if f >= start_f:
                p1, p2 = edges[i]['p1'], edges[i]['p2']
                dist = edges[i]['dist']
                mid = (p1 + p2) / 2
                norm = get_normal(p1, p2)
                
                # Check normal direction, ensure it points outward
                if np.dot(norm, mid - center) < 0:
                    norm = -norm
                    
                tpos = mid + norm * 55
                
                # Determine how much of the line to draw
                if f < start_f + frames_per_edge - 1:
                    frac = (f - start_f + 1) / float(frames_per_edge)
                else:
                    frac = 1.0
                    
                # In frame 17+, only the longest edge's line/text is drawn
                if f >= 17 and i != longest_edge_idx:
                    continue
                    
                # After frame 20, no lines or text are drawn (only the final circle)
                if f >= 21:
                    continue
                
                color = (255, 150, 0) # BGR (Light Blue/Cyan)
                if f >= 17 and i == longest_edge_idx:
                    color = (0, 0, 255) # BGR (Red)
                
                cur_p2 = p1 + frac * (p2 - p1)
                cv2.line(img, tuple(p1.astype(int)), tuple(cur_p2.astype(int)), color, 8)
                
                if frac == 1.0:
                    text = f"{int(dist)}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.2
                    thickness = 3
                    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                    text_x = int(tpos[0] - text_size[0] / 2)
                    text_y = int(tpos[1] + text_size[1] / 2)
                    
                    t_color = (0, 0, 0) # Black text
                    if f >= 17 and i == longest_edge_idx:
                        t_color = (0, 0, 255) # Red text
                    
                    # Draw text with white outline for visibility
                    cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness + 5)
                    cv2.putText(img, text, (text_x, text_y), font, font_scale, t_color, thickness)
        
        # Mark the midpoint of the longest side with a red circle
        if f >= 19:
            mid = (edges[longest_edge_idx]['p1'] + edges[longest_edge_idx]['p2']) / 2
            r = 6 if f == 19 else 12
            cv2.circle(img, tuple(mid.astype(int)), r, (0, 0, 255), -1)
            
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
    os.makedirs('/app/output', exist_ok=True)
    imageio.mimsave(
        '/app/output/video.mp4', 
        frames, 
        fps=16, 
        macro_block_size=None, 
        codec='libx264', 
        pixelformat='yuv420p'
    )

if __name__ == '__main__':
    main()
