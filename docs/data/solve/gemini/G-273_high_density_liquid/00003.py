import cv2
import numpy as np
import math
import os
import subprocess

def main():
    os.makedirs('/app/frames', exist_ok=True)
    os.makedirs('/app/output', exist_ok=True)
    
    img = cv2.imread('/app/first_frame.png')
    
    obj_0 = img[94:159, 236:301].copy()
    obj_1 = img[94:159, 480:545].copy()
    obj_2 = img[94:159, 724:789].copy()
    
    bg = img.copy()
    bg[94:159, 236:301] = [255, 255, 255]
    bg[94:159, 480:545] = [255, 255, 255]
    bg[94:159, 724:789] = [255, 255, 255]
    
    def get_y_obj0(f):
        if f < 10:
            return 94
        t = f - 10
        if t <= 26:
            return 94 + 409 * (t / 26)**2
        elif t <= 36:
            tp = (t - 26) / 10
            return 503 + 47 * math.sin(tp * math.pi / 2)
        elif t <= 46:
            tb = (t - 36) / 10
            return 550 - 14 * math.sin(tb * math.pi / 2)
        else:
            return 536

    def get_y_obj1(f):
        if f < 10:
            return 94
        t = f - 10
        if t <= 23:
            return 94 + 320 * (t / 23)**2
        else:
            ts = t - 23
            y = 414 + 8.5 * ts
            return min(y, 791)

    def get_y_obj2(f):
        if f < 10:
            return 94
        t = f - 10
        if t <= 24:
            return 94 + 348 * (t / 24)**2
        else:
            ts = t - 24
            y = 442 + 8.5 * ts
            return min(y, 791)

    for f in range(80):
        frame = bg.copy()
        
        y0 = int(round(get_y_obj0(f)))
        y1 = int(round(get_y_obj1(f)))
        y2 = int(round(get_y_obj2(f)))
        
        frame[y0:y0+65, 236:301] = obj_0
        frame[y1:y1+65, 480:545] = obj_1
        frame[y2:y2+65, 724:789] = obj_2
        
        cv2.imwrite(f'/app/frames/frame_{f:04d}.png', frame)
        
    cmd = [
        'ffmpeg', '-y', '-framerate', '16', '-i', '/app/frames/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ]
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
