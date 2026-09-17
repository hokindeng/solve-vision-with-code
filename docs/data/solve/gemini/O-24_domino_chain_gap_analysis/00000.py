import cv2
import numpy as np
import math
import os
import subprocess

def get_thetaA_from_thetaB(thB, D, w=43, h=141):
    val = (w - D * math.cos(thB)) / h
    if val > 1.0: val = 1.0
    if val < -1.0: val = -1.0
    return thB - math.asin(val)

def ease_in(x):
    if x <= 0: return 0
    if x >= 1: return 1
    return x**2

def main():
    w = 43
    h = 141
    D12 = 261 - 185
    D23 = 320 - 261

    frames_count = 53
    angles = []
    
    for i in range(frames_count):
        t = i / (frames_count - 1)
        th3 = (math.pi/2) * ease_in((t - 0.25) / 0.75)
        th2 = (53.93 * math.pi/180) * ease_in((t - 0.15) / 0.75)
        th1 = (54.6 * math.pi/180) * ease_in(t / 0.75)
        
        if th3 >= math.pi/2:
            limit2 = math.atan2(D23, w)
        else:
            limit2 = get_thetaA_from_thetaB(th3, D23, w, h)
        if th2 > limit2: th2 = limit2
        
        limit1 = get_thetaA_from_thetaB(th2, D12, w, h)
        if th1 > limit1: th1 = limit1
            
        angles.append([th1, th2, th3])
        
    img = cv2.imread('/app/first_frame.png')
    s1 = img[598:739, 143:186].copy()
    s2 = img[598:739, 219:262].copy()
    s3 = img[598:739, 278:321].copy()
    
    def prepare_sprite(s):
        rgba = cv2.cvtColor(s, cv2.COLOR_BGR2BGRA)
        padded = cv2.copyMakeBorder(rgba, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(20, 20, 20, 0))
        return padded

    ps1 = prepare_sprite(s1)
    ps2 = prepare_sprite(s2)
    ps3 = prepare_sprite(s3)
    
    os.makedirs('/app/output', exist_ok=True)
    os.makedirs('/app/tmp_frames', exist_ok=True)
    
    for i in range(frames_count):
        frame_img = img.copy()
        
        # Erase dominos
        frame_img[598:739, 143:321] = [255, 255, 255]
        
        # Restore brown floor line at 737, 738
        frame_img[737:739, 143:321] = [43, 90, 139]
        
        th1, th2, th3 = angles[i]
        
        def draw_domino(base_img, sprite, x0, theta):
            xp = x0 + 42
            yp = 738
            
            T1 = np.array([[1, 0, -43], [0, 1, -141], [0, 0, 1]], dtype=float)
            
            c = math.cos(theta)
            s = math.sin(theta)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
            
            T2 = np.array([[1, 0, xp], [0, 1, yp], [0, 0, 1]], dtype=float)
            
            M = T2 @ R @ T1
            M_affine = M[:2, :]
            
            warped = cv2.warpAffine(sprite, M_affine, (1024, 1024), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
            
            alpha = warped[:, :, 3:] / 255.0
            base_img[:] = base_img * (1 - alpha) + warped[:, :, :3] * alpha

        draw_domino(frame_img, ps3, 278, th3)
        draw_domino(frame_img, ps2, 219, th2)
        draw_domino(frame_img, ps1, 143, th1)
        
        cv2.imwrite(f'/app/tmp_frames/frame_{i:04d}.png', frame_img)
        
    cmd = [
        'ffmpeg', '-y', '-framerate', '16', '-i', '/app/tmp_frames/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ]
    subprocess.run(cmd, check=True)
    
if __name__ == '__main__':
    main()
