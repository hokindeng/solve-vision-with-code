import os
import cv2
import numpy as np
import imageio

def get_largest_angle_vertex(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return (0, 0)
        
    c = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
    
    pts = [pt[0] for pt in approx]
    
    max_angle = -1
    best_vertex = pts[0]
    
    n = len(pts)
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i - 1 + n) % n]
        
        v1 = p2 - p1
        v2 = p3 - p1
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            continue
            
        cos_theta = np.dot(v1, v2) / (norm1 * norm2)
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        
        if angle > max_angle:
            max_angle = angle
            best_vertex = p1
            
    return tuple(int(x) for x in best_vertex)

def make_video():
    first_frame_path = '/app/first_frame.png'
    output_dir = '/app/output'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'video.mp4')
    
    img = cv2.imread(first_frame_path)
    vertex = get_largest_angle_vertex(img)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    num_frames = 22
    fps = 16
    
    writer = imageio.get_writer(
        output_path, 
        fps=fps, 
        codec='libx264', 
        pixelformat='yuv420p',
        macro_block_size=None
    )
    
    radius = 50
    thickness = 8
    color = (255, 0, 0)
    
    for i in range(num_frames):
        frame = img_rgb.copy()
        
        end_angle = int(360 * i / (num_frames - 1))
        
        if end_angle > 0:
            cv2.ellipse(frame, vertex, (radius, radius), 0, 0, end_angle, color, thickness, cv2.LINE_AA)
            
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    make_video()
