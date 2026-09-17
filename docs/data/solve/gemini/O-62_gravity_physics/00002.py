import cv2
import numpy as np
import os
import subprocess

def get_state(t):
    if t == 0:
        return 11.2, 3.6
        
    t_curr = 0.0
    y_curr = 11.2
    v_curr = 3.6
    
    # Check if t is within the first flight
    dt_impact = (3.6 + (3.6**2 + 20*11.2)**0.5) / 10
    if t < dt_impact:
        y = y_curr + v_curr * t - 5 * t**2
        v = v_curr - 10 * t
        return y, v
        
    t_curr += dt_impact
    v_curr = (10 * dt_impact - v_curr) * 0.7 # v after bounce
    
    while True:
        if v_curr < 0.05: # stop completely
            return 0.0, 0.0
            
        dt_flight = 2 * v_curr / 10
        if t < t_curr + dt_flight:
            dt = t - t_curr
            y = v_curr * dt - 5 * dt**2
            v = v_curr - 10 * dt
            return y, v
            
        t_curr += dt_flight
        v_curr = v_curr * 0.7

def draw_arrow(img, pt1, pt2, color, thickness=2, tip_len=10, tip_width=6):
    cv2.line(img, pt1, pt2, color, thickness)
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    L = (dx**2 + dy**2)**0.5
    if L == 0: return
    dx /= L; dy /= L
    tip_base_x = pt2[0] - tip_len * dx
    tip_base_y = pt2[1] - tip_len * dy
    nx = -dy; ny = dx
    p1 = (int(round(tip_base_x + tip_width * nx)), int(round(tip_base_y + tip_width * ny)))
    p2 = (int(round(tip_base_x - tip_width * nx)), int(round(tip_base_y - tip_width * ny)))
    cv2.fillPoly(img, [np.array([pt2, p1, p2])], color)

def main():
    img = cv2.imread('/app/first_frame.png')
    ball_img = img[357:414, 533:590].copy()
    mask = np.any(ball_img != 255, axis=-1)
    
    bg = img.copy()
    bg[350:420, 520:600] = [255, 255, 255]
    bg[320:355, 360:575] = [255, 255, 255]
    
    os.makedirs('/app/output', exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = '/app/output/temp.mp4'
    writer = cv2.VideoWriter(out_path, fourcc, 16.0, (1024, 1024))
    
    total_frames = 192
    fps = 16.0
    
    for f in range(total_frames):
        t = f / fps
        y, v = get_state(t)
        
        y_bottom_px = 771.4 - y * 32
        y_top = int(round(y_bottom_px - 56))
        x_left = 533
        
        frame = bg.copy()
        
        y1, y2 = max(0, y_top), min(1024, y_top+57)
        x1, x2 = max(0, x_left), min(1024, x_left+57)
        
        by1 = 0 if y_top >= 0 else -y_top
        by2 = 57 if y_top+57 <= 1024 else 1024 - y_top
        bx1 = 0 if x_left >= 0 else -x_left
        bx2 = 57 if x_left+57 <= 1024 else 1024 - x_left
        
        if y2 > y1 and x2 > x1:
            roi = frame[y1:y2, x1:x2]
            b_roi = ball_img[by1:by2, bx1:bx2]
            m_roi = mask[by1:by2, bx1:bx2]
            roi[m_roi] = b_roi[m_roi]
            frame[y1:y2, x1:x2] = roi
            
        y_center = y_top + 28.5
        
        if abs(v) >= 0.1:
            length = int(round(abs(v) * 6.111))
            if v > 0:
                y_base = int(round(y_center - 35))
                y_tip = int(round(y_base - length))
            else:
                y_base = int(round(y_center + 35))
                y_tip = int(round(y_base + length))
                
            draw_arrow(frame, (561, y_base), (561, y_tip), (60, 180, 60), 2, 10, 6)
            
            text = f"v = {abs(v):.1f} m/s"
            size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            text_x = 505 - size[0]
            text_y = int(round((y_base + y_tip) / 2 + size[1] / 2 - 4))
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 180, 60), 2)
            
        writer.write(frame)
        
    writer.release()
    
    subprocess.run([
        'ffmpeg', '-y', '-i', '/app/output/temp.mp4', 
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', 
        '/app/output/video.mp4'
    ], check=True, capture_output=True)
    os.remove('/app/output/temp.mp4')

if __name__ == '__main__':
    main()
