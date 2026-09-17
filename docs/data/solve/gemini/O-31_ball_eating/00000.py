import numpy as np
from PIL import Image, ImageDraw
import imageio
import os

def generate_video():
    os.makedirs('/app/output', exist_ok=True)

    balls_info = {
        'pink': {'center': (562, 78), 'r': 20, 'c': (255, 105, 180)},
        'green': {'center': (403, 849), 'r': 28, 'c': (60, 179, 113)},
        'orange': {'center': (650, 92), 'r': 50, 'c': (255, 140, 0)},
        'crimson': {'center': (469, 360), 'r': 75, 'c': (220, 20, 60)}
    }

    sequence = ['pink', 'green', 'orange', 'crimson']
    start_center = (150, 191)
    start_r = 24

    # Load initial frame and create a clean background
    img0 = Image.open('/app/first_frame.png').convert('RGB')
    bg = img0.copy()
    draw_bg = ImageDraw.Draw(bg)
    
    # Erase the starting black ball
    draw_bg.ellipse([start_center[0]-start_r, start_center[1]-start_r, 
                     start_center[0]+start_r, start_center[1]+start_r], fill=(255, 255, 255))
                     
    # Erase all colored balls
    for name, b in balls_info.items():
        x, y = b['center']
        r = b['r']
        draw_bg.ellipse([x-r, y-r, x+r, y+r], fill=(255, 255, 255))

    frames = []

    # Segments with frame counts
    segments = [
        (20, 'pink'),
        (36, 'green'),
        (37, 'orange'),
        (14, 'crimson')
    ]

    alive = set(sequence)
    current_center = start_center
    current_r = start_r

    # Frame 0 (initial state)
    frames.append(np.array(img0))

    # Animate each segment
    for seg_frames, target_name in segments:
        target = balls_info[target_name]
        target_center = target['center']
        
        for i in range(1, seg_frames + 1):
            t = i / seg_frames
            x = current_center[0] + (target_center[0] - current_center[0]) * t
            y = current_center[1] + (target_center[1] - current_center[1]) * t
            
            if i == seg_frames:
                alive.remove(target_name)
                current_r += target['r']
                current_center = target_center
                
            frame = bg.copy()
            draw = ImageDraw.Draw(frame)
            
            # Draw alive balls
            for name in alive:
                b = balls_info[name]
                bx, by = b['center']
                br = b['r']
                draw.ellipse([bx-br, by-br, bx+br, by+br], fill=b['c'])
                
            # Draw the black ball
            draw.ellipse([x-current_r, y-current_r, x+current_r, y+current_r], fill=(0, 0, 0))
            
            frames.append(np.array(frame))

    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    generate_video()
