import numpy as np
import cv2
from PIL import Image
import os
import subprocess
import shutil

def solve():
    img = Image.open('/app/first_frame.png')
    arr = np.array(img)

    def get_tile(x, y):
        return arr[y:y+215, x:x+215].copy()

    tl_orig = get_tile(280, 280)
    tr_orig = get_tile(530, 280)
    bl_orig = get_tile(280, 530)
    br_orig = get_tile(530, 530)

    def rotate(tile, angle):
        M = cv2.getRotationMatrix2D((107.0, 107.0), angle, 1.0)
        return cv2.warpAffine(tile, M, (215, 215), flags=cv2.INTER_NEAREST, borderValue=(255, 255, 255))

    # Rotations needed to reach the solved circle state (CCW)
    rot_tl = 360.0
    rot_tr = 192.0
    rot_bl = 81.0
    rot_br = 207.0

    os.makedirs('/app/tmp_frames', exist_ok=True)
    os.makedirs('/app/output', exist_ok=True)

    for i in range(96):
        progress = i / 95.0
        frame = arr.copy()
        
        frame[280:280+215, 280:280+215] = rotate(tl_orig, rot_tl * progress)
        frame[280:280+215, 530:530+215] = rotate(tr_orig, rot_tr * progress)
        frame[530:530+215, 280:280+215] = rotate(bl_orig, rot_bl * progress)
        frame[530:530+215, 530:530+215] = rotate(br_orig, rot_br * progress)
        
        Image.fromarray(frame).save(f'/app/tmp_frames/frame_{i:04d}.png')

    subprocess.run([
        'ffmpeg', '-y', '-framerate', '16', '-i', '/app/tmp_frames/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    shutil.rmtree('/app/tmp_frames')

if __name__ == '__main__':
    solve()
