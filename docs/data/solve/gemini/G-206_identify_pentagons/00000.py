import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    original = img.copy()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pentagon = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        if peri < 100 or cv2.contourArea(c) < 1000:
            continue
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 5:
            pentagon = approx
            break

    if pentagon is None:
        print("Pentagon not found")
        return

    M = cv2.moments(pentagon)
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])
    
    max_dist = 0
    for p in pentagon:
        dist = np.sqrt((p[0][0] - cX)**2 + (p[0][1] - cY)**2)
        if dist > max_dist:
            max_dist = dist
            
    target_radius = int(np.ceil(max_dist) + 5)
    
    num_frames = 30
    fps = 16
    
    os.makedirs('/app/output', exist_ok=True)
    out_path = '/app/output/video.mp4'
    writer = imageio.get_writer(out_path, fps=fps, codec='libx264', pixelformat='yuv420p')
    
    for i in range(num_frames):
        frame = original.copy()
        if i > 0:
            r = int((i / (num_frames - 1)) * target_radius)
            # Draw a red circle outline
            cv2.circle(frame, (cX, cY), r, (0, 0, 255), 6, cv2.LINE_AA)
            
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    solve()
