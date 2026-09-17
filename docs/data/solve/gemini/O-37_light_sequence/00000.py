import cv2
import numpy as np
import os
import subprocess

def main():
    img = cv2.imread('/app/first_frame.png')
    
    y_c = 512
    x_centers = [142, 224, 306, 388, 470, 552, 634, 716, 798, 880]
    
    # Target off patch from light 1 (which is off)
    x_off = x_centers[0]
    patch_off = img[y_c-41:y_c+41, x_off-41:x_off+41].astype(np.float32)
    
    num_frames = 35
    os.makedirs('/app/output', exist_ok=True)
    
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', '1024x1024',
        '-pix_fmt', 'bgr24',
        '-r', '16',
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '/app/output/video.mp4'
    ]
    
    # redirect stdout and stderr to DEVNULL to keep output clean,
    # or just let it print
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    for i in range(num_frames):
        t = i / (num_frames - 1)
        frame = img.copy()
        
        # Lights 7 and 9 (indices 6 and 8) need to turn off
        for idx in [6, 8]:
            x_c = x_centers[idx]
            patch_current = img[y_c-41:y_c+41, x_c-41:x_c+41].astype(np.float32)
            
            blended = patch_current * (1 - t) + patch_off * t
            frame[y_c-41:y_c+41, x_c-41:x_c+41] = np.clip(blended, 0, 255).astype(np.uint8)
            
        process.stdin.write(frame.tobytes())
        
    process.stdin.close()
    process.wait()

if __name__ == '__main__':
    main()
