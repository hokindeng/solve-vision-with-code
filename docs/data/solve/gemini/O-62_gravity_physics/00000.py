import cv2
import numpy as np
import imageio

def solve():
    img = cv2.imread('/app/first_frame.png')
    
    # Extract ball
    ball_patch = img[238:295, 239:296].copy()
    mask = np.any(ball_patch != 255, axis=-1)
    
    # Create clean background
    clean_bg = img.copy()
    clean_bg[150:773, 230:480] = 255
    
    # Physics parameters
    y = 295.0
    v = 57.6
    g = 220.8
    y_ground = 773.0
    elasticity = 0.70
    dt = 1/16.0
    
    x_center = 267.5
    y0_center = 295.0 - 28.5
    max_y_center = y0_center
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    for frame_idx in range(192):
        frame = clean_bg.copy()
        
        y_center = y - 28.5
        max_y_center = max(max_y_center, y_center)
        
        # Draw trajectory line
        cv2.line(frame, (int(x_center), int(y0_center)), (int(x_center), int(max_y_center)), (200, 200, 200), 2, lineType=cv2.LINE_AA)
        
        # Paste ball
        y_start = int(y - 57)
        x_start = int(x_center - 28.5)
        for c in range(3):
            frame[y_start:y_start+57, x_start:x_start+57, c] = np.where(
                mask, ball_patch[:,:,c], frame[y_start:y_start+57, x_start:x_start+57, c]
            )
            
        # Draw velocity arrow and text
        v_m = abs(v) / 32.0
        arrow_len = int(v_m * 12.222)
        
        if arrow_len > 0:
            if v > 0: # down
                pt1 = (int(x_center), int(y))
                pt2 = (int(x_center), int(y) + arrow_len)
            else: # up
                pt1 = (int(x_center), int(y - 57))
                pt2 = (int(x_center), int(y - 57) - arrow_len)
                
            tip_length = min(0.5, 10.0 / arrow_len) if arrow_len > 0 else 0.5
            cv2.arrowedLine(frame, pt1, pt2, (60, 180, 60), 3, tipLength=tip_length, line_type=cv2.LINE_AA)
            
            text_x = int(x_center) + 15
            text_y = (pt1[1] + pt2[1]) // 2 + 5
            cv2.putText(frame, f"v = {v_m:.1f} m/s", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 180, 60), 2, lineType=cv2.LINE_AA)
        else:
            text_x = int(x_center) + 15
            text_y = int(y_center) + 5
            cv2.putText(frame, f"v = 0.0 m/s", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 180, 60), 2, lineType=cv2.LINE_AA)

        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Physics update
        if y >= y_ground - 1e-3 and abs(v) < 5.0:
            y = y_ground
            v = 0
            continue
            
        t_remaining = dt
        bounce_count = 0
        while t_remaining > 0 and bounce_count < 20:
            a = 0.5 * g
            b = v
            c = y - y_ground
            
            D = b**2 - 4*a*c
            hit_ground = False
            t_hit = float('inf')
            
            if D >= 0:
                t1 = (-b + np.sqrt(D)) / (2*a)
                t2 = (-b - np.sqrt(D)) / (2*a)
                valid_t = [t for t in (t1, t2) if t > 1e-7]
                if valid_t:
                    t_hit = min(valid_t)
                    if t_hit <= t_remaining:
                        hit_ground = True
            
            if hit_ground:
                y = y_ground
                v = v + g * t_hit
                v = -v * elasticity
                if abs(v) < 5.0:
                    v = 0
                    y = y_ground
                    break
                t_remaining -= t_hit
                bounce_count += 1
            else:
                y = y + v * t_remaining + 0.5 * g * t_remaining**2
                v = v + g * t_remaining
                t_remaining = 0
                
        if bounce_count >= 20:
            y = y_ground
            v = 0

    writer.close()

if __name__ == '__main__':
    solve()
