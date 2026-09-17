import cv2
import numpy as np
import os
import imageio

def generate_video():
    y0 = 12.5
    v0 = -0.9
    g = 10.1
    e = 0.70

    # calculate bounce times and velocities
    bounces = []
    a = 0.5 * (-g)
    b = v0
    c = y0
    disc = b**2 - 4*a*c
    t1 = (-b - np.sqrt(disc)) / (2*a)
    v1 = abs(v0 + (-g)*t1)

    t_curr = 0.0
    bounces.append({
        't_start': t_curr,
        't_end': t1,
        'y0': y0,
        'v0': v0
    })
    t_curr = t1
    v_bounce = v1 * e

    while v_bounce > 0.01:
        dt = 2 * v_bounce / g
        bounces.append({
            't_start': t_curr,
            't_end': t_curr + dt,
            'y0': 0.0,
            'v0': v_bounce
        })
        t_curr += dt
        v_bounce *= e

    def get_state(t):
        if t >= t_curr:
            return 0.0, 0.0
        for b in bounces:
            if b['t_start'] <= t < b['t_end']:
                dt = t - b['t_start']
                y = b['y0'] + b['v0']*dt - 0.5*g*dt**2
                v = b['v0'] - g*dt
                return max(0.0, y), v
        return 0.0, 0.0

    img = cv2.imread('/app/first_frame.png')
    
    # Extract ball patch exactly using bounding box found previously
    y_min, y_max = 304, 370
    x_min, x_max = 402, 468
    
    ball_patch = img[y_min:y_max+1, x_min:x_max+1].copy()
    
    # Create exact alpha mask
    alpha_mask = np.any(ball_patch != 255, axis=-1).astype(np.float32)
    
    cx = int(np.mean([x_min, x_max]))
    cy = int(np.mean([y_min, y_max]))
    
    H, W, _ = ball_patch.shape
    
    # Clean background by filling the ball's location with white
    bg = img.copy()
    bg[y_min:y_max+1, x_min:x_max+1] = [255, 255, 255]

    # Scale
    ground_y = 773
    radius = (y_max - y_min) / 2.0
    center_y_ground = ground_y - radius
    center_y_start = cy
    scale = (center_y_ground - center_y_start) / 12.5

    frames = []
    fps = 16
    duration = 12.0
    total_frames = int(fps * duration)

    trail_points = []
    
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, codec='libx264', format='FFMPEG', pixelformat='yuv420p', macro_block_size=None)

    for i in range(total_frames):
        t = i / fps
        y_m, v_m = get_state(t)
        
        # Calculate pixel position
        curr_cy = center_y_ground - y_m * scale
        
        # We want to record every frame's position for the trail
        trail_points.append(curr_cy)
        
        frame = bg.copy()
        
        # Draw trail
        if len(trail_points) > 1:
            pts = np.array([[cx, int(round(ty))] for ty in trail_points], np.int32)
            pts = pts.reshape((-1, 1, 2))
            # draw a faint line for the trajectory
            cv2.polylines(frame, [pts], isClosed=False, color=(180, 180, 180), thickness=2)
            
        # Draw ball
        top_y = int(round(curr_cy - H/2))
        bottom_y = top_y + H
        left_x = int(round(cx - W/2))
        right_x = left_x + W
        
        if top_y >= 0 and bottom_y <= frame.shape[0] and left_x >= 0 and right_x <= frame.shape[1]:
            roi = frame[top_y:bottom_y, left_x:right_x]
            alpha_3d = alpha_mask[:, :, np.newaxis]
            blended = (ball_patch * alpha_3d + roi * (1 - alpha_3d)).astype(np.uint8)
            frame[top_y:bottom_y, left_x:right_x] = blended
        
        # Draw velocity arrow
        v_scale = 5.0
        v_px = -v_m * v_scale 
        
        if abs(v_px) > 2:
            end_y = int(round(curr_cy + v_px))
            tip = min(0.3, 10.0 / abs(v_px))
            cv2.arrowedLine(frame, (cx, int(round(curr_cy))), (cx, end_y), (0, 0, 255), 3, tipLength=tip)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)

    writer.close()
    print("Video generation complete.")

if __name__ == '__main__':
    generate_video()
