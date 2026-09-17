import cv2
import numpy as np
import imageio
import os

def remove_color(img, color):
    mask = np.all(img == color, axis=-1)
    res = img.copy()
    res[mask] = [255, 255, 255]
    return res

def draw_black_ball(img, center, radius):
    shift = 4
    factor = 1 << shift
    cx = int(round(center[0] * factor))
    cy = int(round(center[1] * factor))
    r = int(round(radius * factor))
    cv2.circle(img, (cx, cy), r, (0, 0, 0), -1, cv2.LINE_8, shift=shift)

def get_state(frame):
    if frame <= 30:
        progress = frame / 30.0
        x = 678 + (97 - 678) * progress
        y = 320 + (696 - 320) * progress
        return (x, y), 28.0, ['orange', 'pink', 'green']
        
    elif frame <= 32:
        progress = (frame - 30) / 2.0
        r = 28.0 + (48.0 - 28.0) * progress
        return (97, 696), r, ['pink', 'green']
        
    elif frame <= 72:
        progress = (frame - 32) / 40.0
        x = 97 + (917 - 97) * progress
        y = 696 + (309 - 696) * progress
        return (x, y), 48.0, ['pink', 'green']
        
    elif frame <= 75:
        progress = (frame - 72) / 3.0
        r = 48.0 + (86.0 - 48.0) * progress
        return (917, 309), r, ['green']
        
    elif frame <= 90:
        progress = (frame - 75) / 15.0
        x = 917 + (955 - 917) * progress
        y = 309 + (655 - 309) * progress
        return (x, y), 86.0, ['green']
        
    elif frame <= 94:
        progress = (frame - 90) / 4.0
        r = 86.0 + (144.0 - 86.0) * progress
        return (955, 655), r, []
        
    else:
        return (955, 655), 144.0, []

def main():
    os.makedirs('/app/output', exist_ok=True)
    img = cv2.imread('/app/first_frame.png')
    
    bg_all = remove_color(img, [0, 0, 0])
    
    balls_info = {
        'orange': [0, 140, 255],
        'pink': [180, 105, 255],
        'green': [113, 179, 60]
    }
    
    out_frames = []
    
    for f in range(95):
        if f == 0:
            frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out_frames.append(frame_rgb)
            continue
            
        b_pos, b_r, visible = get_state(f)
        
        frame_img = bg_all.copy()
        
        if 'orange' not in visible:
            frame_img = remove_color(frame_img, balls_info['orange'])
        if 'pink' not in visible:
            frame_img = remove_color(frame_img, balls_info['pink'])
        if 'green' not in visible:
            frame_img = remove_color(frame_img, balls_info['green'])
            
        draw_black_ball(frame_img, b_pos, b_r)
        
        frame_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        out_frames.append(frame_rgb)
        
    imageio.mimwrite(
        '/app/output/video.mp4', 
        out_frames, 
        fps=16, 
        codec='libx264', 
        pixelformat='yuv420p',
        quality=9,
        macro_block_size=1
    )

if __name__ == '__main__':
    main()
