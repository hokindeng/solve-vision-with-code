import cv2
import numpy as np
import imageio
import os

def main():
    img = cv2.imread('/app/first_frame.png')
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    crosses = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        M = cv2.moments(c)
        if M['m00'] == 0: continue
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        
        bx = x + w/2.0
        by = y + h/2.0
        
        dir_x = cx - bx
        dir_y = cy - by
        
        dx = cx - 512
        dy = cy - 512
        
        cross = dx * dir_y - dy * dir_x
        crosses.append((c, cross, (x, y, w, h)))
        
    pos_count = sum(1 for _, cr, _ in crosses if cr > 0)
    neg_count = sum(1 for _, cr, _ in crosses if cr < 0)
    
    target_contour = None
    if pos_count == 1:
        target_contour = next(c for c, cr, _ in crosses if cr > 0)
    elif neg_count == 1:
        target_contour = next(c for c, cr, _ in crosses if cr < 0)
    else:
        target_contour = contours[0] # Fallback
        
    tx, ty, tw, th = cv2.boundingRect(target_contour)
    center_x = tx + tw // 2
    center_y = ty + th // 2
    
    radius = int(max(tw, th) * 0.7)
    if radius < 20: radius = 100
    
    frames = []
    num_frames = 48
    
    for i in range(num_frames):
        frame = img.copy()
        angle = int((i / (num_frames - 1)) * 360)
        
        if angle > 0:
            cv2.ellipse(frame, (center_x, center_y), (radius, radius), 0, -90, angle - 90, (0, 0, 255), 8, cv2.LINE_AA)
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    for f in frames:
        writer.append_data(f)
    writer.close()

if __name__ == '__main__':
    main()
