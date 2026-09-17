import cv2
import numpy as np
import subprocess
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    bg_color = img[0, 0]

    diff = np.abs(img.astype(np.int32) - bg_color.astype(np.int32))
    mask = np.sum(diff, axis=2) > 0
    mask_uint8 = (mask * 255).astype(np.uint8)

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objects = []
    baseline = None
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 500 and h < 20:
            baseline = {'x': x, 'y': y, 'w': w, 'h': h}
        elif w > 10 and h > 10:
            obj_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.drawContours(obj_mask, [c], -1, 255, -1)
            sprite = img[y:y+h, x:x+w].copy()
            sprite_mask = obj_mask[y:y+h, x:x+w].copy()
            objects.append({
                'start_x': x,
                'start_y': y,
                'w': w,
                'h': h,
                'sprite': sprite,
                'mask': sprite_mask
            })

    # Sort objects by area (w * h) from largest to smallest
    objects.sort(key=lambda obj: obj['w'] * obj['h'], reverse=True)

    # Calculate end positions
    total_w = sum(obj['w'] for obj in objects)
    available_w = baseline['w']
    gaps_count = len(objects) + 1
    gap = (available_w - total_w) / gaps_count

    for i, obj in enumerate(objects):
        exact_x = baseline['x'] + gap * (i + 1) + sum(objects[j]['w'] for j in range(i))
        obj['end_x'] = int(round(exact_x))
        obj['end_y'] = baseline['y'] - obj['h']

    # Create clean background
    clean_bg = img.copy()
    for obj in objects:
        mask_bool = obj['mask'] > 0
        for c in range(3):
            clean_bg[obj['start_y']:obj['start_y']+obj['h'], obj['start_x']:obj['start_x']+obj['w'], c] = \
                np.where(mask_bool, bg_color[c], clean_bg[obj['start_y']:obj['start_y']+obj['h'], obj['start_x']:obj['start_x']+obj['w'], c])

    def ease_in_out(t):
        return t * t * (3 - 2 * t)

    num_frames = 40
    fps = 16
    out_path = '/app/output/video.mp4'
    
    os.makedirs('/app/output', exist_ok=True)
    
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{img.shape[1]}x{img.shape[0]}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        out_path
    ]
    
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for i in range(num_frames):
        t = i / (num_frames - 1)
        progress = ease_in_out(t)
        
        frame = clean_bg.copy()
        
        for obj in objects:
            curr_x = int(round(obj['start_x'] + (obj['end_x'] - obj['start_x']) * progress))
            curr_y = int(round(obj['start_y'] + (obj['end_y'] - obj['start_y']) * progress))
            
            y1, y2 = curr_y, curr_y + obj['h']
            x1, x2 = curr_x, curr_x + obj['w']
            
            mask_bool = obj['mask'] > 0
            for c in range(3):
                frame[y1:y2, x1:x2, c] = np.where(mask_bool, obj['sprite'][:, :, c], frame[y1:y2, x1:x2, c])
                
        process.stdin.write(frame.tobytes())
        
    process.stdin.close()
    process.wait()

if __name__ == '__main__':
    solve()
