import cv2
import numpy as np
from PIL import Image, ImageDraw
import imageio
import os

def main():
    img = cv2.imread('/app/first_frame.png')
    
    clean_img = img.copy()
    for y in range(500, 608):
        for x in range(500, 750):
            c = img[y, x]
            if not np.array_equal(c, [255,255,255]) and not np.array_equal(c, [255,0,0]) and not np.array_equal(c, [150,150,150]):
                clean_img[y, x] = [255, 255, 255]

    bg = Image.fromarray(cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)).convert('RGBA')

    start_pt = (564, 609)
    end_pt = (1024, 419)
    color = (0, 0, 255, int(255 * 0.38))
    
    num_frames = 35
    frames = []

    for i in range(num_frames):
        progress = i / (num_frames - 1)
        
        curr_x = start_pt[0] + (end_pt[0] - start_pt[0]) * progress
        curr_y = start_pt[1] + (end_pt[1] - start_pt[1]) * progress
        
        overlay = Image.new('RGBA', (1024, 1024), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        if progress > 0:
            draw.line([start_pt, (curr_x, curr_y)], fill=color, width=3)
            
            angle = np.arctan2(-233, 564)
            length = 30
            width = 12
            
            tip = (curr_x, curr_y)
            base_cx = curr_x - length * np.cos(angle)
            base_cy = curr_y - length * np.sin(angle)
            
            p1 = (base_cx + width * np.sin(angle), base_cy - width * np.cos(angle))
            p2 = (base_cx - width * np.sin(angle), base_cy + width * np.cos(angle))
            
            dist = np.hypot(curr_x - start_pt[0], curr_y - start_pt[1])
            if dist > length:
                draw.polygon([tip, p1, p2], fill=color)
            else:
                scale = dist / length
                if scale > 0:
                    base_cx_s = curr_x - (length * scale) * np.cos(angle)
                    base_cy_s = curr_y - (length * scale) * np.sin(angle)
                    p1_s = (base_cx_s + (width * scale) * np.sin(angle), base_cy_s - (width * scale) * np.cos(angle))
                    p2_s = (base_cx_s - (width * scale) * np.sin(angle), base_cy_s + (width * scale) * np.cos(angle))
                    draw.polygon([tip, p1_s, p2_s], fill=color)
                
        out = Image.alpha_composite(bg, overlay).convert('RGB')
        frames.append(np.array(out))

    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=1, quality=10)

if __name__ == '__main__':
    main()
