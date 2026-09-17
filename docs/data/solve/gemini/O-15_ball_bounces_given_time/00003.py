import os
import cv2
import numpy as np
import subprocess

def create_video():
    os.makedirs('/app/output', exist_ok=True)
    os.makedirs('/app/frames', exist_ok=True)
    
    img = cv2.imread('/app/first_frame.png')
    bg = img.copy()
    
    # Define colors (in BGR)
    cyan = [255, 191, 0]
    orange = [0, 140, 255]
    white = [255, 255, 255]
    
    # Create clean background
    cyan_mask = (bg == cyan).all(axis=-1)
    bg[cyan_mask] = white
    
    arrow_mask = (img == orange).all(axis=-1)
    
    # Physics parameters
    x0, y0 = 515.5, 893.5
    vx, vy = 1.0, -0.7001113
    
    # Wall inner bounds are 63 to 961, ball radius is 30
    xmin, xmax = 93.0, 931.0
    ymin, ymax = 93.0, 931.0
    
    x, y = x0, y0
    cvx, cvy = vx, vy
    events = [(0, x, y, cvx, cvy)]
    t_total = 0
    
    # Simulate 3 collisions
    for _ in range(3):
        tx = (xmax - x) / cvx if cvx > 0 else (xmin - x) / cvx if cvx < 0 else float('inf')
        ty = (ymax - y) / cvy if cvy > 0 else (ymin - y) / cvy if cvy < 0 else float('inf')
        
        t = min(tx, ty)
        x += cvx * t
        y += cvy * t
        t_total += t
        
        if tx < ty:
            cvx = -cvx
        else:
            cvy = -cvy
            
        events.append((t_total, x, y, cvx, cvy))
        
    frames = 80
    dt_frame = t_total / (frames - 1)
    
    for f in range(frames):
        if f == 0:
            frame = img.copy()
        else:
            t_target = f * dt_frame
            
            fx, fy = x0, y0
            for i in range(len(events)-1):
                t1, x1, y1, vx1, vy1 = events[i]
                t2, x2, y2, vx2, vy2 = events[i+1]
                
                # Add small epsilon to handle float precision for last frame
                if t1 <= t_target <= t2 + 1e-5:
                    dt = t_target - t1
                    fx = x1 + vx1 * dt
                    fy = y1 + vy1 * dt
                    break
                    
            frame = bg.copy()
            # Draw the ball
            cv2.circle(frame, (int(round(fx)), int(round(fy))), 30, cyan, -1)
            # Ensure the arrow stays exactly as in the first frame
            frame[arrow_mask] = orange
            
        cv2.imwrite(f'/app/frames/frame_{f:04d}.png', frame)
        
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-framerate', '16',
        '-i', '/app/frames/frame_%04d.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '/app/output/video.mp4'
    ]
    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == '__main__':
    create_video()
