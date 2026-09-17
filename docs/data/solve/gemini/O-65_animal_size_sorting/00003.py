import cv2
import numpy as np
import imageio

def ease_in_out_quad(t):
    if t < 0.5:
        return 2 * t * t
    else:
        return -1 + (4 - 2 * t) * t

def main():
    img = cv2.imread('/app/first_frame.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    clean_bg = img.copy()
    objects = []

    baseline_y = 944
    baseline_start_x = 30
    baseline_end_x = 995

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 900: 
            continue # skip baseline
        
        mask = thresh[y:y+h, x:x+w] > 0
        patch = img[y:y+h, x:x+w].copy()
        
        objects.append({
            'patch': patch,
            'mask': mask,
            'start_x': x,
            'start_y': y,
            'w': w,
            'h': h,
            'area': w * h
        })
        
        clean_bg[y:y+h, x:x+w][mask] = [255, 255, 255]

    # Sort objects by size (area) from largest to smallest
    objects.sort(key=lambda obj: obj['area'], reverse=True)

    # Calculate target positions
    num_objects = len(objects)
    total_width = sum(obj['w'] for obj in objects)
    available_space = (baseline_end_x - baseline_start_x) - total_width
    gap = available_space / (num_objects - 1) if num_objects > 1 else 0

    current_x = baseline_start_x
    for obj in objects:
        obj['target_x'] = int(round(current_x))
        obj['target_y'] = baseline_y - obj['h']
        current_x += obj['w'] + gap

    # Generate frames
    num_frames = 40
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)

    for frame_idx in range(num_frames):
        # Calculate interpolation factor
        if num_frames > 1:
            t = frame_idx / (num_frames - 1)
        else:
            t = 1.0
        
        # Apply easing
        progress = ease_in_out_quad(t)
        
        frame = clean_bg.copy()
        
        for obj in objects:
            curr_x = int(round(obj['start_x'] + (obj['target_x'] - obj['start_x']) * progress))
            curr_y = int(round(obj['start_y'] + (obj['target_y'] - obj['start_y']) * progress))
            
            h, w = obj['h'], obj['w']
            
            # Blend object into frame
            roi = frame[curr_y:curr_y+h, curr_x:curr_x+w]
            roi[obj['mask']] = obj['patch'][obj['mask']]
            frame[curr_y:curr_y+h, curr_x:curr_x+w] = roi

        # imageio expects RGB, OpenCV uses BGR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)

    writer.close()

if __name__ == '__main__':
    main()
