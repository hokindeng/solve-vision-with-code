import cv2
import numpy as np
import subprocess
import os

def ease_in_out(t):
    return -0.5 * (np.cos(np.pi * t) - 1.0)

def main():
    img = cv2.imread('/app/first_frame.png')
    
    is_bg = (img[:,:,0] == 235) & (img[:,:,1] == 235) & (img[:,:,2] == 235)
    
    bg_frame = img.copy()
    bg_frame[~is_bg] = [235, 235, 235]
    
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[~is_bg] = 255
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shapes = []
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        
        sprite_mask = mask[y:y+h, x:x+w]
        sprite = img[y:y+h, x:x+w]
        
        pixels = sprite.reshape(-1, 3)
        bg_mask_flat = sprite_mask.reshape(-1)
        fg_pixels = [tuple(p) for p, m in zip(pixels, bg_mask_flat) if m > 0 and tuple(p) != (80, 80, 80)]
        
        from collections import Counter
        if fg_pixels:
            main_color = Counter(fg_pixels).most_common(1)[0][0]
        else:
            main_color = (0,0,0)
            
        shapes.append({
            'start_x': x,
            'start_y': y,
            'w': w,
            'h': h,
            'color': main_color,
            'sprite': sprite.copy(),
            'mask': sprite_mask > 0
        })
        
    shapes.sort(key=lambda s: (s['color'], s['w']))
    
    total_w = sum(s['w'] for s in shapes)
    gap = (1024 - total_w) / (len(shapes) + 1)
    
    current_x = gap
    for s in shapes:
        s['end_x'] = int(round(current_x))
        s['end_y'] = int(round(512 - s['h'] / 2.0))
        current_x += s['w'] + gap
        
    os.makedirs('/app/output', exist_ok=True)
    
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', '1024x1024',
        '-pix_fmt', 'bgr24',
        '-r', '16',
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '/app/output/video.mp4'
    ]
    
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    
    num_frames = 96
    for frame_idx in range(num_frames):
        t = frame_idx / max(1, (num_frames - 1))
        alpha = ease_in_out(t)
        
        frame = bg_frame.copy()
        
        for s in shapes:
            curr_x = int(round(s['start_x'] + alpha * (s['end_x'] - s['start_x'])))
            curr_y = int(round(s['start_y'] + alpha * (s['end_y'] - s['start_y'])))
            
            h, w = s['h'], s['w']
            frame_slice = frame[curr_y:curr_y+h, curr_x:curr_x+w]
            np.copyto(frame_slice, s['sprite'], where=s['mask'][:,:,None])
            
        process.stdin.write(frame.tobytes())
        
    process.stdin.close()
    process.wait()

if __name__ == '__main__':
    main()
