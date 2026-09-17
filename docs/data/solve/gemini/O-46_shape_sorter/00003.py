import cv2
import numpy as np
import math
import imageio
import os

def main():
    img = cv2.imread('/app/first_frame.png')
    
    colors = {
        'orange': [60, 146, 251],
        'yellow': [21, 204, 250],
        'blue': [250, 165, 96],
        'cyan': [238, 211, 34]
    }
    
    moves = {
        'yellow': (492, -8),
        'blue': (492, -10),
        'cyan': (488, 5),
        'orange': (490, 5)
    }
    
    order = ['yellow', 'blue', 'cyan', 'orange']
    
    shapes_masks = {}
    clean_bg = img.copy()
    bg_color = np.array([252, 250, 248])
    
    for name in order:
        color = colors[name]
        mask = cv2.inRange(img, np.array(color), np.array(color))
        shapes_masks[name] = mask
        clean_bg[mask > 0] = bg_color
        
    frames = []
    total_frames = 80
    
    # Let's allocate 18 frames per movement (0 to 17)
    move_duration = 18
    
    for frame_idx in range(total_frames):
        frame = clean_bg.copy()
        
        draw_list = []
        moving_shape = None
        moving_dx = 0
        moving_dy = 0
        
        for i, name in enumerate(order):
            start_f = i * move_duration
            end_f = start_f + move_duration - 1
            
            final_dx, final_dy = moves[name]
            
            if frame_idx < start_f:
                dx, dy = 0, 0
                draw_list.append((name, dx, dy))
            elif frame_idx > end_f:
                dx, dy = final_dx, final_dy
                draw_list.append((name, dx, dy))
            else:
                p = (frame_idx - start_f) / (end_f - start_f)
                eased_p = 0.5 - 0.5 * math.cos(math.pi * p)
                dx = int(round(final_dx * eased_p))
                dy = int(round(final_dy * eased_p))
                moving_shape = name
                moving_dx = dx
                moving_dy = dy
                
        # Draw stationary shapes first
        for name, dx, dy in draw_list:
            mask = shapes_masks[name]
            color = colors[name]
            y_idx, x_idx = np.where(mask > 0)
            new_y = y_idx + dy
            new_x = x_idx + dx
            valid = (new_y >= 0) & (new_y < frame.shape[0]) & (new_x >= 0) & (new_x < frame.shape[1])
            frame[new_y[valid], new_x[valid]] = color
            
        # Draw moving shape on top
        if moving_shape:
            mask = shapes_masks[moving_shape]
            color = colors[moving_shape]
            y_idx, x_idx = np.where(mask > 0)
            new_y = y_idx + moving_dy
            new_x = x_idx + moving_dx
            valid = (new_y >= 0) & (new_y < frame.shape[0]) & (new_x >= 0) & (new_x < frame.shape[1])
            frame[new_y[valid], new_x[valid]] = color
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, format='FFMPEG', codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    main()
