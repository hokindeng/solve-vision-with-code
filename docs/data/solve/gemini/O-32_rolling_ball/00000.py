import cv2
import numpy as np
import imageio

def solve():
    img = cv2.imread('/app/first_frame.png')
    
    ball_mask = ((img[:,:,1] > 150) & (img[:,:,2] > 150) & (img[:,:,0] < 50)).astype(np.uint8) * 255
    x, y, w, h = cv2.boundingRect(ball_mask)
    
    ball_img = img[y:y+h, x:x+w].copy()
    ball_alpha = ball_mask[y:y+h, x:x+w]
    
    base_img = img.copy()
    base_img[y:y+h, x:x+w][ball_alpha == 255] = [255, 255, 255]
    
    plat_mask = ((img[:,:,0] > 100) & (img[:,:,1] > 50) & (img[:,:,2] < 150)).astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(plat_mask, connectivity=8)
    
    platforms = []
    for i in range(1, num_labels):
        bx, by, bw, bh, area = stats[i]
        platforms.append((bx, by, bw, bh))
        
    platforms = sorted(platforms, key=lambda p: p[0])
    
    key_points = []
    start_x = x + w / 2.0
    start_y = y + h
    key_points.append((start_x, start_y))
    
    for bx, by, bw, bh in platforms:
        cx = bx + bw / 2.0
        ty = by
        key_points.append((cx, ty))
        
    frames = []
    total_frames = 64
    move_frames = 60
    
    for f in range(total_frames):
        if f >= move_frames:
            cx, cy = key_points[-1]
        else:
            t = f / float(move_frames)
            target_cx = key_points[0][0] + t * (key_points[-1][0] - key_points[0][0])
            
            # Find which segment
            for i in range(len(key_points) - 1):
                x0, y0 = key_points[i]
                x1, y1 = key_points[i+1]
                
                if target_cx <= x1 + 1e-5:
                    u = (target_cx - x0) / (x1 - x0) if x1 > x0 else 0
                    base_y = y0 + u * (y1 - y0)
                    H = 45 if i == 0 else 15
                    jump_y = H * 4 * u * (1 - u)
                    
                    cx = target_cx
                    cy = base_y - jump_y
                    break
                    
        frame = base_img.copy()
        bx = int(round(cx - w / 2.0))
        by = int(round(cy - h))
        
        roi = frame[by:by+h, bx:bx+w]
        for c in range(3):
            roi[:, :, c] = np.where(ball_alpha == 255, ball_img[:, :, c], roi[:, :, c])
            
        frame[by:by+h, bx:bx+w] = roi
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    import os
    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    solve()
