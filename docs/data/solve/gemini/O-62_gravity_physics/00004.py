import cv2
import numpy as np
import subprocess
import os

def draw_arrow(img, pt1, pt2, color, thickness=6):
    length = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
    if length < 1:
        return
    tipLength = 15.0 / length if length > 15 else 1.0
    cv2.arrowedLine(img, pt1, pt2, color, thickness, tipLength=tipLength)

def main():
    first_frame = cv2.imread('/app/first_frame.png')
    
    # Physics parameters
    y0_phys = 13.3
    v0_phys = 0.7
    g = 6.1
    e = 0.70
    
    scale_y = 32.0 
    scale_v = 61.0 / 0.7 
    
    segments = []
    t_curr = 0.0
    y_curr = y0_phys
    v_curr = v0_phys
    
    for _ in range(50):
        a = -0.5 * g
        b = v_curr
        c = y_curr
        disc = b**2 - 4*a*c
        if disc < 0: disc = 0
        dt = (-b - np.sqrt(disc)) / (2*a)
        
        segments.append({
            't_start': t_curr,
            't_end': t_curr + dt,
            'y_start': y_curr,
            'v_start': v_curr
        })
        
        v_impact = v_curr - g * dt
        y_curr = 0.0
        v_curr = -v_impact * e
        t_curr += dt
        
        if v_curr < 0.01:
            break
            
    fps = 16
    total_frames = 192
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('/app/temp.mp4', fourcc, fps, (1024, 1024))
    
    bg = first_frame.copy()
    ball_mask = cv2.inRange(first_frame, np.array([70, 190, 50]), np.array([90, 210, 70]))
    kernel = np.ones((5,5), np.uint8)
    ball_mask_dilated = cv2.dilate(ball_mask, kernel, iterations=1)
    bg = cv2.inpaint(bg, ball_mask_dilated, 3, cv2.INPAINT_TELEA)
    
    coords = np.argwhere(ball_mask > 0)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    ball_sprite = first_frame[y_min:y_max+1, x_min:x_max+1].copy()
    sprite_mask = ball_mask[y_min:y_max+1, x_min:x_max+1]
    
    sprite_center_y = ball_sprite.shape[0] // 2
    sprite_center_x = ball_sprite.shape[1] // 2
    
    ball_x = 727
    
    trail = []
    
    for frame_idx in range(total_frames):
        if frame_idx == 0:
            out.write(first_frame)
            trail.append((ball_x, 313))
            continue
            
        t = frame_idx / fps
        
        y_phys = 0.0
        v_phys = 0.0
        
        if t >= segments[-1]['t_end']:
            y_phys = 0.0
            v_phys = 0.0
        else:
            for seg in segments:
                if seg['t_start'] <= t <= seg['t_end']:
                    dt = t - seg['t_start']
                    y_phys = seg['y_start'] + seg['v_start'] * dt - 0.5 * g * dt**2
                    v_phys = seg['v_start'] - g * dt
                    break
                    
        if y_phys < 0: y_phys = 0
        
        # 313 is the initial center y when y_phys is 13.3
        # 313 = ground_y_center - 13.3 * 32.0 => ground_y_center = 738.6
        current_cy = int(round(738.6 - y_phys * scale_y))
        
        frame = bg.copy()
        
        trail.append((ball_x, current_cy))
        
        if len(trail) > 1:
            for i in range(1, len(trail)):
                cv2.line(frame, trail[i-1], trail[i], (80, 200, 60), 2)
                
        y1 = current_cy - sprite_center_y
        y2 = y1 + ball_sprite.shape[0]
        x1 = ball_x - sprite_center_x
        x2 = x1 + ball_sprite.shape[1]
        
        for c in range(3):
            frame[y1:y2, x1:x2, c] = np.where(sprite_mask > 0, 
                                              ball_sprite[:, :, c], 
                                              frame[y1:y2, x1:x2, c])
                                              
        if abs(v_phys) > 0.05:
            arrow_len = abs(v_phys) * scale_v
            pt1 = (ball_x, current_cy)
            if v_phys > 0:
                pt2 = (ball_x, int(round(current_cy - arrow_len)))
            else:
                pt2 = (ball_x, int(round(current_cy + arrow_len)))
                
            draw_arrow(frame, pt1, pt2, (220, 60, 60), thickness=6)
            
        out.write(frame)
        
    out.release()
    
    os.makedirs('/app/output', exist_ok=True)
    subprocess.run([
        'ffmpeg', '-y', '-i', '/app/temp.mp4', 
        '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', 
        '/app/output/video.mp4'
    ], check=True)

if __name__ == '__main__':
    main()
