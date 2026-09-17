import cv2
import numpy as np
import imageio
import os

def create_video():
    img = cv2.imread('/app/first_frame.png')
    
    # Target positions based on liquid properties
    # Cup 1, 2, 3 have higher density -> object floats -> target_y rests on surface
    # Cup 4 has lighter color -> lower density -> object sinks -> target_y rests on cup bottom
    objs_info = [
        {'x': 130, 'y': 94, 'w': 67, 'h': 67, 'target_y': 496},  # Cup 1 (surface at 563)
        {'x': 374, 'y': 94, 'w': 67, 'h': 67, 'target_y': 547},  # Cup 2 (surface at 614)
        {'x': 618, 'y': 94, 'w': 67, 'h': 67, 'target_y': 529},  # Cup 3 (surface at 596)
        {'x': 862, 'y': 94, 'w': 67, 'h': 67, 'target_y': 790}   # Cup 4 (bottom at 857)
    ]
    
    # Extract objects
    objects = []
    for info in objs_info:
        x, y, w, h = info['x'], info['y'], info['w'], info['h']
        obj_img = img[y:y+h, x:x+w].copy()
        objects.append(obj_img)
        
    # Create clean background by removing objects
    clean_bg = img.copy()
    for info in objs_info:
        x, y, w, h = info['x'], info['y'], info['w'], info['h']
        clean_bg[y:y+h, x:x+w] = [255, 255, 255]
        
    os.makedirs('/app/output', exist_ok=True)
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    
    total_frames = 80
    start_move_frame = 8
    end_move_frame = 72
    
    for frame in range(total_frames):
        current_frame = clean_bg.copy()
        
        if frame <= start_move_frame:
            t = 0.0
        elif frame >= end_move_frame:
            t = 1.0
        else:
            # Linear interpolation fraction
            linear_t = (frame - start_move_frame) / (end_move_frame - start_move_frame)
            # Smoothstep easing
            t = linear_t * linear_t * (3.0 - 2.0 * linear_t)
            
        for i, info in enumerate(objs_info):
            curr_y = int(info['y'] + t * (info['target_y'] - info['y']))
            x, w, h = info['x'], info['w'], info['h']
            
            # Draw the object
            current_frame[curr_y:curr_y+h, x:x+w] = objects[i]
            
        # Convert BGR to RGB before writing to video
        current_frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        writer.append_data(current_frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    create_video()
