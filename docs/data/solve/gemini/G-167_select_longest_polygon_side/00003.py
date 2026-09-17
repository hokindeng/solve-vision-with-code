import cv2
import numpy as np
import imageio
import os

def generate_video():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("Could not read /app/first_frame.png")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    points = approx.reshape(-1, 2).tolist()
    n_points = len(points)
    
    lengths = []
    for i in range(n_points):
        p1 = points[i]
        p2 = points[(i + 1) % n_points]
        lengths.append(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        
    frames = []
    
    # Frame 0: Original
    frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    current_best = -1
    
    # Frames 1-16
    for i in range(n_points):
        # Frame 1 of this edge: checking (YELLOW)
        frame1 = img.copy()
        
        # Draw current best in GREEN (if any and not the current edge)
        if current_best != -1 and current_best != i:
            cb_p1 = points[current_best]
            cb_p2 = points[(current_best + 1) % n_points]
            cv2.line(frame1, tuple(cb_p1), tuple(cb_p2), (0, 255, 0), 6) # Green
            
        # Draw current being checked in YELLOW
        cp1 = points[i]
        cp2 = points[(i + 1) % n_points]
        cv2.line(frame1, tuple(cp1), tuple(cp2), (0, 255, 255), 6) # Yellow
        frames.append(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        
        # Frame 2 of this edge: result of comparison
        frame2 = img.copy()
        if current_best == -1 or lengths[i] > lengths[current_best]:
            current_best = i
            
        # Draw new/current best in GREEN
        cb_p1 = points[current_best]
        cb_p2 = points[(current_best + 1) % n_points]
        cv2.line(frame2, tuple(cb_p1), tuple(cb_p2), (0, 255, 0), 6) # Green
        frames.append(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
            
    # Longest edge is current_best
    longest_idx = current_best
    p1 = points[longest_idx]
    p2 = points[(longest_idx + 1) % n_points]
    midpoint = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
    
    # Frames 17-24: Red circle at midpoint
    for i in range(8):
        frame = img.copy()
        
        # Grow the circle over the first 4 frames
        radius = min(3 + i * 3, 12)
        
        cv2.circle(frame, midpoint, radius, (0, 0, 255), -1) # Red
        
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    for f in frames:
        writer.append_data(f)
    writer.close()

if __name__ == "__main__":
    generate_video()
