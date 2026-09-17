import cv2
import numpy as np
import imageio
import os

def make_video():
    first_frame = cv2.imread('/app/first_frame.png')
    
    # Identify innermost square
    unique_colors = np.unique(first_frame.reshape(-1, first_frame.shape[2]), axis=0)
    squares = []
    for color in unique_colors:
        mask = cv2.inRange(first_frame, color, color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if abs(w - h) < 5 and w > 10 and w < first_frame.shape[1] - 5:
                squares.append({'x': x, 'y': y, 'w': w, 'h': h})
                
    if not squares:
        print("No squares found!")
        return
        
    squares.sort(key=lambda s: s['w'])
    innermost = squares[0]
    
    x, y, w, h = innermost['x'], innermost['y'], innermost['w'], innermost['h']
    
    thickness = 10
    half_t = thickness // 2
    
    # Outer and inner bounds of the outline
    x1_out, y1_out = x - half_t, y - half_t
    x2_out, y2_out = x + w + half_t, y + h + half_t
    
    x1_in, y1_in = x + half_t, y + half_t
    x2_in, y2_in = x + w - half_t, y + h - half_t
    
    L_w = x2_out - x1_out
    L_h = y2_out - y1_out
    
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    full_outline = np.zeros_like(first_frame_rgb)
    
    # Draw top, right, bottom, left with RGB Blue (0, 0, 255)
    full_outline[y1_out:y1_in, x1_out:x2_out] = [0, 0, 255]
    full_outline[y1_out:y2_out, x2_in:x2_out] = [0, 0, 255]
    full_outline[y2_in:y2_out, x1_out:x2_out] = [0, 0, 255]
    full_outline[y1_out:y2_out, x1_out:x1_in] = [0, 0, 255]
    
    total_frames = 85
    start_frame = 10
    end_frame = 75
    
    total_path = 2 * L_w + 2 * L_h
    
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', format='FFMPEG', pixelformat='yuv420p')
    
    for i in range(total_frames):
        img = first_frame_rgb.copy()
        
        if i >= start_frame:
            progress = min(1.0, (i - start_frame) / (end_frame - start_frame))
            d = int(progress * total_path)
            
            mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
            
            rem = d
            if rem > 0:
                l = min(rem, L_w)
                mask[y1_out:y1_in, x1_out:x1_out+l] = 1
                rem -= l
            if rem > 0:
                l = min(rem, L_h)
                mask[y1_out:y1_out+l, x2_in:x2_out] = 1
                rem -= l
            if rem > 0:
                l = min(rem, L_w)
                mask[y2_in:y2_out, x2_out-l:x2_out] = 1
                rem -= l
            if rem > 0:
                l = min(rem, L_h)
                mask[y2_out-l:y2_out, x1_out:x1_in] = 1
                
            img = np.where(mask == 1, full_outline, img)
            
        writer.append_data(img)
        
    writer.close()

if __name__ == '__main__':
    make_video()
