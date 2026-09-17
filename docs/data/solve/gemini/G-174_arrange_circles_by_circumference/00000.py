import cv2
import numpy as np
import imageio
import os

def ease_in_out(t):
    return t * t * (3.0 - 2.0 * t)

def main():
    img = cv2.imread('/app/first_frame.png')
    bg_color = img[0, 0].tolist()

    diff = cv2.absdiff(img, np.full_like(img, bg_color, dtype=np.uint8))
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circles = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        sprite = img[y:y+h, x:x+w].copy()
        
        s_mask = np.any(sprite != bg_color, axis=-1)
        
        circles.append({
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + w / 2.0,
            'cy': y + h / 2.0,
            'sprite': sprite,
            'mask': s_mask,
            'circumference': w
        })

    circles.sort(key=lambda c: c['circumference'], reverse=True)

    gap = 20
    total_width = sum(c['w'] for c in circles) + gap * (len(circles) - 1)
    start_x = (img.shape[1] - total_width) / 2.0
    target_y = img.shape[0] / 2.0

    current_x = start_x
    for c in circles:
        c['target_cx'] = current_x + c['w'] / 2.0
        c['target_cy'] = target_y
        current_x += c['w'] + gap

    os.makedirs('/app/output', exist_ok=True)
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None, quality=10)

    num_frames = 80
    for i in range(num_frames):
        t = i / (num_frames - 1)
        t_eased = ease_in_out(t)
        
        frame = np.full_like(img, bg_color, dtype=np.uint8)
        
        for c in reversed(circles):
            cx = c['cx'] + (c['target_cx'] - c['cx']) * t_eased
            cy = c['cy'] + (c['target_cy'] - c['cy']) * t_eased
            
            x = int(round(cx - c['w'] / 2.0))
            y = int(round(cy - c['h'] / 2.0))
            
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img.shape[1], x + c['w']), min(img.shape[0], y + c['h'])
            
            sx1, sy1 = x1 - x, y1 - y
            sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
            
            if x2 > x1 and y2 > y1:
                patch = frame[y1:y2, x1:x2]
                s = c['sprite'][sy1:sy2, sx1:sx2]
                m = c['mask'][sy1:sy2, sx1:sx2]
                
                patch[m] = s[m]
                
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)

    writer.close()

if __name__ == '__main__':
    main()
