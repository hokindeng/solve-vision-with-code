import cv2
import numpy as np
import subprocess
import os

def create_video():
    img_first = cv2.imread('/app/first_frame.png')
    
    # 1. Define crop areas for circles and texts
    circle_boxes = {
        'N': (440, 150, 580, 280),
        'S': (440, 730, 580, 870),
        'W': (150, 440, 290, 580),
        'E': (730, 440, 880, 580)
    }
    
    text_boxes = {
        'N': (470, 290, 550, 385),
        'S': (470, 875, 550, 970),
        'W': (180, 580, 260, 675),
        'E': (760, 580, 840, 675)
    }
    
    # 2. Extract circle masks
    circle_masks = {}
    for key, (x1, y1, x2, y2) in circle_boxes.items():
        crop = img_first[y1:y2, x1:x2]
        # Yellow is [0, 200, 255] in BGR
        mask = cv2.inRange(crop, np.array([0, 190, 240]), np.array([10, 210, 255]))
        circle_masks[key] = mask
        
    # 3. Get text crops
    original_text_crops = {
        1: img_first[290:385, 470:550].copy(),
        2: img_first[875:970, 470:550].copy(),
        4: img_first[580:675, 760:840].copy()
    }
    
    crop3 = np.full((95, 80, 3), 255, dtype=np.uint8)
    text = "3"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2.8
    thick = 8
    text_size = cv2.getTextSize(text, font, scale, thick)[0]
    cx = 40
    cy = 47
    tx = cx - text_size[0] // 2
    ty = cy + text_size[1] // 2
    cv2.putText(crop3, text, (tx, ty), font, scale, (0, 0, 0), thick, cv2.LINE_AA)
    original_text_crops[3] = crop3
    
    # Colors (BGR)
    colors = {
        'R': (0, 0, 255),
        'Y': (0, 200, 255),
        'G': (0, 255, 0)
    }
    
    cycle = ['R', 'Y', 'G', 'Y']
    
    initial_states = {
        'N': (1, 1.0),
        'S': (1, 2.0),
        'E': (3, 4.0),
        'W': (3, 1.0)
    }
    
    def get_light_state(start_idx, start_time, t):
        if t < start_time:
            current_idx = start_idx
            rem = start_time - t
        else:
            t_after = t - start_time
            phases_passed = int(t_after // 4) + 1
            current_idx = (start_idx + phases_passed) % 4
            rem = 4.0 - (t_after % 4.0)
        
        if rem == 0.0:
            rem = 4.0
            
        number = int(np.ceil(rem))
        number = max(1, min(4, number))
        return cycle[current_idx], number

    os.makedirs('/app/output', exist_ok=True)
    out_path = '/app/output/video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 16
    total_frames = 112
    writer = cv2.VideoWriter(out_path, fourcc, fps, (1024, 1024))
    
    for i in range(total_frames):
        t = min(i / fps, 6.0)
        
        frame = img_first.copy()
        
        for key in ['N', 'S', 'E', 'W']:
            c_char, num = get_light_state(initial_states[key][0], initial_states[key][1], t)
            
            # update circle color
            cx1, cy1, cx2, cy2 = circle_boxes[key]
            crop_c = frame[cy1:cy2, cx1:cx2]
            c_mask = circle_masks[key]
            crop_c[c_mask > 127] = colors[c_char]
            frame[cy1:cy2, cx1:cx2] = crop_c
            
            # update text
            tx1, ty1, tx2, ty2 = text_boxes[key]
            frame[ty1:ty2, tx1:tx2] = original_text_crops[num]
            
        writer.write(frame)
        
    writer.release()
    
    subprocess.run([
        'ffmpeg', '-y', '-i', out_path,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '/app/output/video_final.mp4'
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.rename('/app/output/video_final.mp4', out_path)

if __name__ == '__main__':
    create_video()
