import cv2
import numpy as np
import os
import subprocess

def draw_partial(img, x1, y1, x2, y2, t, L):
    if L > 0:
        d = min(L, x2 - x1)
        cv2.rectangle(img, (x1 - t, y1 - t), (x1 + d + t, y1 + t), (0, 255, 0), -1)
        L -= d
    if L > 0:
        d = min(L, y2 - y1)
        cv2.rectangle(img, (x2 - t, y1 - t), (x2 + t, y1 + d + t), (0, 255, 0), -1)
        L -= d
    if L > 0:
        d = min(L, x2 - x1)
        cv2.rectangle(img, (x2 + t, y2 - t), (x2 - d - t, y2 + t), (0, 255, 0), -1)
        L -= d
    if L > 0:
        d = min(L, y2 - y1)
        cv2.rectangle(img, (x1 - t, y2 + t), (x1 + t, y2 - d - t), (0, 255, 0), -1)
        L -= d

def main():
    img0 = cv2.imread('/app/first_frame.png')
    x1, y1, x2, y2 = 698, 61, 965, 310
    total_L = (x2 - x1)*2 + (y2 - y1)*2
    frames = 28
    
    os.makedirs('/app/output', exist_ok=True)
    
    for i in range(frames):
        img = img0.copy()
        L = int((i / (frames - 1)) * total_L)
        draw_partial(img, x1, y1, x2, y2, 4, L)
        cv2.imwrite(f'/app/output/frame_{i:04d}.png', img)
        
    subprocess.run([
        'ffmpeg', '-y', '-framerate', '16', '-i', '/app/output/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ], check=True)
    
    for i in range(frames):
        try:
            os.remove(f'/app/output/frame_{i:04d}.png')
        except:
            pass

if __name__ == '__main__':
    main()
