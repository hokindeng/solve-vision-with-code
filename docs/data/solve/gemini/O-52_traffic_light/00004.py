import cv2
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont
import math
import os

def get_state(initial_phase, initial_countdown, sim_time):
    # phases: 0=Red, 1=Yellow, 2=Green, 3=Yellow2
    phases = [("Red", 4), ("Yellow", 4), ("Green", 4), ("Yellow", 4)]
    
    t0 = 0
    for i, (p, dur) in enumerate(phases):
        if i == initial_phase:
            t0 += (dur - initial_countdown)
            break
        t0 += dur
        
    t = (t0 + sim_time) % 16
    
    curr_t = 0
    for i, (p, dur) in enumerate(phases):
        if curr_t <= t < curr_t + dur:
            rem = (curr_t + dur) - t
            cnt = math.ceil(rem)
            if cnt == 0: cnt = 1
            return p, cnt
        curr_t += dur
    return "Red", 1

bounds = {
    'N': (400, 100, 200, 200),
    'S': (400, 700, 200, 200),
    'E': (700, 400, 200, 200),
    'W': (100, 400, 200, 200)
}
bgr = {
    'Red': [0, 0, 255],
    'Yellow': [0, 200, 255],
    'Green': [0, 255, 0]
}

text_bounds = {
    'N': (451, 276, 123, 123),
    'S': (451, 860, 123, 123),
    'E': (743, 568, 123, 123),
    'W': (159, 568, 123, 123)
}

font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
try:
    font = ImageFont.truetype(font_path, 102)
except IOError:
    font = ImageFont.load_default()

def set_light(img, pos, color_name):
    x, y, w, h = bounds[pos]
    roi = img[y:y+h, x:x+w]
    mask = cv2.inRange(roi, np.array([0, 0, 255]), np.array([0, 0, 255]))
    mask |= cv2.inRange(roi, np.array([0, 200, 255]), np.array([0, 200, 255]))
    mask |= cv2.inRange(roi, np.array([0, 255, 0]), np.array([0, 255, 0]))
    
    roi[mask > 0] = bgr[color_name]

def set_text(img, pos, num):
    x, y, w, h = text_bounds[pos]
    
    box_img = Image.new('RGB', (w, h), 'white')
    draw = ImageDraw.Draw(box_img)
    
    bbox = font.getbbox(str(num))
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (w - tw) / 2 - bbox[0]
    ty = (h - th) / 2 - bbox[1]
    draw.text((tx, ty), str(num), font=font, fill='black')
    
    box_cv = cv2.cvtColor(np.array(box_img), cv2.COLOR_RGB2BGR)
    img[y:y+h, x:x+w] = box_cv


def main():
    os.makedirs('/app/output', exist_ok=True)
    first_frame = cv2.imread('/app/first_frame.png')
    
    initial_states = {
        'N': (0, 4),
        'S': (1, 4),
        'E': (1, 1),
        'W': (1, 3)
    }
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    
    num_frames = 112
    for i in range(num_frames):
        sim_time = i * (6.0 / (num_frames - 1))
        
        img = first_frame.copy()
        for pos, (ip, ic) in initial_states.items():
            color, cnt = get_state(ip, ic, sim_time)
            set_light(img, pos, color)
            set_text(img, pos, cnt)
            
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        writer.append_data(rgb)
        
    writer.close()
    print("Done")

if __name__ == '__main__':
    main()
