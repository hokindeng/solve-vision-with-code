import cv2
import numpy as np
import imageio
import os

def solve():
    # Load original and background
    first_frame = cv2.imread('/app/first_frame_copy.png')
    if first_frame is None:
        first_frame = cv2.imread('/app/first_frame.png')
        
    bg = first_frame.copy()
    x, y_box, w, h = 680, 55, 73, 73
    
    # Extract ball
    ball = bg[y_box:y_box+h, x:x+w].copy()
    gray = cv2.cvtColor(ball, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
    ball_rgba = cv2.cvtColor(ball, cv2.COLOR_BGR2BGRA)
    ball_rgba[:, :, 3] = mask
    
    # Clean background
    bg[y_box:y_box+h, x:x+w] = (255, 255, 255)
    
    # Physics parameters
    y_m = 20.2
    v_m = 0.0
    g = 6.0
    e = 0.70
    dt = 1/16.0
    
    frames = []
    
    def draw_frame(is_first_frame, current_y, current_v):
        if is_first_frame:
            # strictly use original first frame for ball position
            # return exactly the first frame with no modifications
            return first_frame.copy()
            
        frame = bg.copy()
        # 1m = 32 pixels
        y_bottom = 776 - int(round(current_y * 32))
        current_y_box = y_bottom - 73
        
        # Composite ball
        alpha = ball_rgba[:, :, 3] / 255.0
        for c in range(3):
            frame[current_y_box:current_y_box+73, x:x+73, c] = (
                alpha * ball_rgba[:, :, c] +
                (1 - alpha) * frame[current_y_box:current_y_box+73, x:x+73, c]
            )
        cy = current_y_box + 36
        cx = x + 36
            
        # Draw velocity arrow
        scale_v = 10.0 # visually clear length
        arrow_len = int(round(abs(current_v) * scale_v))
        
        color = (0, 0, 255)
        thickness = 3
        
        if arrow_len >= 2:
            end_y = cy - int(round(current_v * scale_v))
            tip = min(0.3, 15.0 / arrow_len)
            cv2.arrowedLine(frame, (cx, cy), (cx, end_y), color, thickness, tipLength=tip)
            
            text = f"{abs(current_v):.1f} m/s"
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # text position
            text_y = cy + (end_y - cy)//2
            cv2.putText(frame, text, (cx + 45, text_y + 5), font, 0.7, color, 2, cv2.LINE_AA)
        else:
            text = "0.0 m/s"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, text, (cx + 45, cy + 5), font, 0.7, color, 2, cv2.LINE_AA)
            
        return frame

    # Simulation loop
    max_frames = 400
    for i in range(max_frames):
        # Draw current state
        frame = draw_frame(i == 0, y_m, v_m)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Advance physics
        t_rem = dt
        y_next_final = y_m
        v_next_final = v_m
        
        while t_rem > 0:
            y_next = y_next_final + v_next_final * t_rem - 0.5 * g * t_rem**2
            if y_next < 0:
                # hit ground
                a = 0.5 * g
                b = -v_next_final
                c = -y_next_final
                disc = b**2 - 4*a*c
                if disc < 0:
                    disc = 0
                t_hit = (-b + np.sqrt(disc)) / (2*a)
                
                v_impact = v_next_final - g * t_hit
                v_next_final = -v_impact * e
                y_next_final = 0.0
                t_rem -= t_hit
                
                if abs(v_next_final) < 0.5 and t_rem > 0:
                    y_next_final = 0.0
                    v_next_final = 0.0
                    break
            else:
                y_next_final = y_next
                v_next_final = v_next_final - g * t_rem
                t_rem = 0
                
        y_m = y_next_final
        v_m = v_next_final
        
        if y_m == 0.0 and v_m == 0.0:
            # add the final frame
            frame = draw_frame(False, y_m, v_m)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            break
            
    # Ensure directory exists
    os.makedirs('/app/output', exist_ok=True)
    
    # Save video
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', macro_block_size=None, pixelformat='yuv420p')
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == "__main__":
    solve()
