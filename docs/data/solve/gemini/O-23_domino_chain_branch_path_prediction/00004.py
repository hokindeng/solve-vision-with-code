import os
import cv2
import numpy as np
import imageio

def get_angle(frame, start, dur, max_a):
    if frame <= start:
        return 0
    if frame >= start + dur:
        return max_a
    t = (frame - start) / dur
    return max_a * (t ** 2)

def generate_video():
    img = cv2.imread('/app/first_frame.png')
    bg_color = np.array([230, 235, 240])

    dominos_config = [
        {'id': 'd0', 'rect': (43, 438, 67, 94), 'start': 0, 'dur': 15, 'max_a': 85},   
        {'id': 'd1', 'rect': (169, 438, 64, 94), 'start': 6, 'dur': 15, 'max_a': 85},  
        {'id': 'd2', 'rect': (294, 438, 64, 94), 'start': 12, 'dur': 15, 'max_a': 85}, 
        {'id': 'd3', 'rect': (419, 438, 64, 94), 'start': 18, 'dur': 15, 'max_a': 85}, 
        
        {'id': 'A0', 'rect': (544, 341, 64, 94), 'start': 24, 'dur': 15, 'max_a': 85}, 
        {'id': 'A1', 'rect': (669, 316, 64, 94), 'start': 30, 'dur': 15, 'max_a': 60}, 
        {'id': 'A2', 'rect': (794, 292, 64, 94), 'start': 999, 'dur': 15, 'max_a': 0}, 
        {'id': 'A3', 'rect': (919, 267, 64, 94), 'start': 999, 'dur': 15, 'max_a': 0}, 
        
        {'id': 'B0', 'rect': (544, 535, 64, 94), 'start': 24, 'dur': 15, 'max_a': 85}, 
        {'id': 'B1', 'rect': (669, 560, 64, 94), 'start': 30, 'dur': 15, 'max_a': 85}, 
        {'id': 'B2', 'rect': (794, 584, 64, 94), 'start': 36, 'dur': 15, 'max_a': 85}, 
        {'id': 'B3', 'rect': (919, 609, 64, 94), 'start': 42, 'dur': 15, 'max_a': 85}  
    ]

    clean_plate = img.copy()
    for d in dominos_config:
        x, y, w, h = d['rect']
        clean_plate[y:y+h, x:x+w] = bg_color

    for d in dominos_config:
        x, y, w, h = d['rect']
        dom = img[y:y+h, x:x+w].copy()
        diff = np.abs(dom.astype(int) - bg_color)
        alpha = np.any(diff > 5, axis=2).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        alpha_filled = np.zeros_like(alpha)
        cv2.drawContours(alpha_filled, contours, -1, 255, -1)
        
        dom_padded = dom.copy()
        dom_padded[alpha_filled == 0] = bg_color
        
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:,:,:3] = dom_padded
        rgba[:,:,3] = alpha_filled
        d['img'] = rgba

    # Sort descending by x
    dominos_config.sort(key=lambda d: d['rect'][0], reverse=True)

    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')

    for frame_idx in range(62):
        if frame_idx == 0:
            writer.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            print("Generated frame 0 (exact)")
            continue
        bg = clean_plate.copy()
        
        for d in dominos_config:
            angle = get_angle(frame_idx, d['start'], d['dur'], d['max_a'])
            x, y, w, h = d['rect']
            rgba = d['img']
            
            if angle == 0:
                alpha_mask = rgba[:,:,3:] / 255.0
                bg[y:y+h, x:x+w] = (bg[y:y+h, x:x+w] * (1 - alpha_mask) + rgba[:,:,:3] * alpha_mask).astype(np.uint8)
            else:
                canvas = np.zeros((1024, 1024, 4), dtype=np.uint8)
                canvas[:,:,:3] = bg_color
                canvas[:,:,3] = 0 
                
                canvas[y:y+h, x:x+w, :3] = rgba[:,:,:3]
                canvas[y:y+h, x:x+w, 3] = rgba[:,:,3]
                
                pivot = (x+w, y+h)
                M = cv2.getRotationMatrix2D(pivot, -angle, 1.0)
                
                rotated = cv2.warpAffine(canvas, M, (1024, 1024))
                
                alpha_mask = rotated[:,:,3:] / 255.0
                bg = (bg * (1 - alpha_mask) + rotated[:,:,:3] * alpha_mask).astype(np.uint8)
                
        writer.append_data(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
        print(f"Generated frame {frame_idx}")
        
    writer.close()

if __name__ == '__main__':
    generate_video()
