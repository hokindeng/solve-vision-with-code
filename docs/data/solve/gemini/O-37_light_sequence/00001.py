import cv2
import numpy as np
import os
import subprocess

def main():
    img = cv2.imread('/app/first_frame.png')
    
    # Get regions
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray < 255
    cols = np.any(mask, axis=0)
    
    from itertools import groupby
    regions = []
    for k, g in groupby(enumerate(cols), key=lambda x: x[1]):
        if k:
            l = list(g)
            regions.append((l[0][0], l[-1][0]))
    centers = [(s + e) / 2 for s, e in regions]
    
    def get_inner_mask(patch):
        mask_off = np.all(patch == [128, 128, 128], axis=-1)
        mask_on = np.all(patch == [0, 215, 255], axis=-1)
        return (mask_off | mask_on).astype(np.float32)

    patch_initial = []
    patch_target = []
    x_starts = []
    
    # Pre-extract all patches
    for i in range(8):
        x_start = int(round(centers[i])) - 50
        x_starts.append(x_start)
        patch = img[462:562, x_start:x_start+100]
        patch_initial.append(patch.copy())
        
    for i in range(8):
        patch = patch_initial[i]
        is_on = np.sum(np.all(patch == [0, 215, 255], axis=-1)) > 50
        target_on = (i % 2 == 0) # 0-indexed: 0, 2, 4, 6 are odd positions -> ON
        
        if is_on == target_on:
            patch_target.append(patch.copy())
        else:
            # Find ref
            ref_idx = -1
            for j in range(8):
                pj = patch_initial[j]
                if (np.sum(np.all(pj == [0, 215, 255], axis=-1)) > 50) == target_on:
                    ref_idx = j
                    break
                    
            ref_patch = patch_initial[ref_idx]
            mask_i = get_inner_mask(patch)
            mask_ref = get_inner_mask(ref_patch)
            
            shift, _ = cv2.phaseCorrelate(mask_i, mask_ref)
            
            M = np.float32([[1, 0, -shift[0]], [0, 1, -shift[1]]])
            shifted_ref = cv2.warpAffine(ref_patch, M, (100, 100), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
            patch_target.append(shifted_ref)
            
    os.makedirs('/app/output', exist_ok=True)
    
    frames_dir = '/app/output/frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    num_frames = 35
    for t in range(num_frames):
        alpha = t / (num_frames - 1)
        frame = img.copy()
        
        for i in range(8):
            p_init = patch_initial[i].astype(np.float32)
            p_targ = patch_target[i].astype(np.float32)
            
            p_t = p_init * (1 - alpha) + p_targ * alpha
            p_t = np.clip(p_t, 0, 255).astype(np.uint8)
            
            xs = x_starts[i]
            frame[462:562, xs:xs+100] = p_t
            
        cv2.imwrite(f'{frames_dir}/frame_{t:04d}.png', frame)
        
    cmd = [
        'ffmpeg', '-y', '-framerate', '16', '-i', f'{frames_dir}/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ]
    subprocess.run(cmd, check=True)
    
if __name__ == '__main__':
    main()
