import cv2
import numpy as np
import os
import subprocess

def main():
    img = cv2.imread('/app/first_frame.png')
    
    # 10 centers, from x=143, spacing=82
    xs = [143 + 82*i for i in range(10)]
    
    def get_patch(idx):
        x = xs[idx]
        return img[512-41:512+41, x-41:x+41].copy()

    gray_target = get_patch(1)
    
    red2 = get_patch(2)
    red5 = get_patch(5)
    red_blend_target = (red2.astype(float) * 0.5 + red5.astype(float) * 0.5).astype(np.uint8)

    num_frames = 35
    fps = 16
    
    output_dir = '/app/output'
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'video.mp4')
    
    # Setup ffmpeg subprocess
    cmd = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{img.shape[1]}x{img.shape[0]}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        out_path
    ]
    
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Store starting patches
    start_patches = {i: get_patch(i) for i in range(10)}
    
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        frame = img.copy()
        
        # turn ON (gray -> red)
        for idx in [0, 1]:
            x = xs[idx]
            patch_start = start_patches[idx]
            patch_current = (patch_start * (1 - alpha) + red_blend_target * alpha).astype(np.uint8)
            frame[512-41:512+41, x-41:x+41] = patch_current
            
        # turn OFF (red -> gray)
        for idx in [4, 5, 6, 8]:
            x = xs[idx]
            patch_start = start_patches[idx]
            patch_current = (patch_start * (1 - alpha) + gray_target * alpha).astype(np.uint8)
            frame[512-41:512+41, x-41:x+41] = patch_current
            
        process.stdin.write(frame.tobytes())
        
    process.stdin.close()
    process.wait()

if __name__ == '__main__':
    main()
