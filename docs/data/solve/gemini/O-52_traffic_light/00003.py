import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio

def get_state(t):
    # returns color, countdown
    t = t % 16.0
    if t < 4: return [0, 0, 255], 4 - int(t % 4)
    elif t < 8: return [0, 200, 255], 4 - int(t % 4)
    elif t < 12: return [0, 200, 0], 4 - int(t % 4)
    else: return [0, 200, 255], 4 - int(t % 4)

def draw_countdown(frame, y1, y2, x1, x2, digit):
    box = frame[y1:y2, x1:x2]
    box[:] = [255, 255, 255]
    pil_box = Image.fromarray(cv2.cvtColor(box, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_box)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 80)
    
    bbox = draw.textbbox((0, 0), str(digit), font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    
    x = (x2 - x1 - w) // 2 - bbox[0]
    y = (y2 - y1 - h) // 2 - bbox[1]
    
    draw.text((x, y), str(digit), font=font, fill=(0, 0, 0))
    box[:] = cv2.cvtColor(np.array(pil_box), cv2.COLOR_RGB2BGR)

def solve():
    os.makedirs('/app/output', exist_ok=True)
    img = cv2.imread('/app/first_frame.png')
    
    w_light_box = (440, 570, 150, 290)
    n_light_box = (150, 280, 440, 580)
    e_light_box = (440, 570, 730, 880)
    
    w_mask = np.all(img[w_light_box[0]:w_light_box[1], w_light_box[2]:w_light_box[3]] == [0, 0, 255], axis=-1)
    n_mask = np.all(img[n_light_box[0]:n_light_box[1], n_light_box[2]:n_light_box[3]] == [0, 200, 255], axis=-1)
    e_mask = np.all(img[e_light_box[0]:e_light_box[1], e_light_box[2]:e_light_box[3]] == [0, 200, 0], axis=-1)
    
    frames = []
    
    for f in range(112):
        sim_t = min(f / 16.0, 6.0)
        
        west_c, west_cd = get_state((0.0 + sim_t) % 16.0)
        north_c, north_cd = get_state((7.0 + sim_t) % 16.0)
        east_c, east_cd = get_state((11.0 + sim_t) % 16.0)
        
        frame = img.copy()
        
        if west_c != [0, 0, 255]:
            frame[w_light_box[0]:w_light_box[1], w_light_box[2]:w_light_box[3]][w_mask] = west_c
        if north_c != [0, 200, 255]:
            frame[n_light_box[0]:n_light_box[1], n_light_box[2]:n_light_box[3]][n_mask] = north_c
        if east_c != [0, 200, 0]:
            frame[e_light_box[0]:e_light_box[1], e_light_box[2]:e_light_box[3]][e_mask] = east_c
            
        if west_cd != 4:
            draw_countdown(frame, 568, 691, 159, 282, west_cd)
        if north_cd != 1:
            draw_countdown(frame, 276, 399, 451, 574, north_cd)
        if east_cd != 1:
            draw_countdown(frame, 568, 691, 743, 866, east_cd)
            
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)

if __name__ == '__main__':
    solve()
