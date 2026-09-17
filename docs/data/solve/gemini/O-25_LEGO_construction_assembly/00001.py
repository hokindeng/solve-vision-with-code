import cv2
import numpy as np
import os
import subprocess

def ease(t):
    return t * t * (3.0 - 2.0 * t)

def main():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("Could not read /app/first_frame.png")
    
    bg = np.array([240, 245, 245])
    diff = np.sum(np.abs(img.astype(int) - bg), axis=2)

    mask = np.zeros_like(diff, dtype=np.uint8)
    mask[(diff > 10) & (np.arange(1024)[None, :] < 350) & (np.arange(1024)[:, None] > 200) & (np.arange(1024)[:, None] < 600)] = 1
    # exclude pure red
    red_pixels = (img[:,:,0] == 0) & (img[:,:,1] == 0) & (img[:,:,2] == 255)
    mask[red_pixels] = 0

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
    brick_label = np.argmax(areas) + 1
    brick_mask = (labels == brick_label)

    # dilate slightly for erasing to ensure no ghost is left in the original spot
    kernel = np.ones((3,3), np.uint8)
    brick_mask_dilated = cv2.dilate(brick_mask.astype(np.uint8), kernel) > 0

    # 2. Create clean background
    clean_bg = img.copy()
    erase_mask = brick_mask_dilated & (~red_pixels)
    clean_bg[erase_mask] = bg

    # 3. Create the blue brick image with alpha (use undilated mask to avoid halo)
    brick_rgba = np.zeros((1024, 1024, 4), dtype=np.uint8)
    brick_rgba[brick_mask] = np.concatenate([img[brick_mask], np.full((np.sum(brick_mask), 1), 255, dtype=np.uint8)], axis=1)
    # just in case
    brick_rgba[red_pixels] = [0,0,0,0]

    os.makedirs('/app/output', exist_ok=True)
    
    # target movement
    target_dx = 256
    target_dy = 440
    num_frames = 46

    out_frames = []

    for i in range(num_frames):
        t = i / (num_frames - 1)
        factor = ease(t)
        
        cur_dx = int(round(target_dx * factor))
        cur_dy = int(round(target_dy * factor))
        
        M = np.float32([[1, 0, cur_dx], [0, 1, cur_dy]])
        shifted_brick = cv2.warpAffine(brick_rgba, M, (1024, 1024), flags=cv2.INTER_NEAREST)
        
        frame = clean_bg.copy()
        
        # alpha blending
        alpha = (shifted_brick[:, :, 3] / 255.0)[:, :, np.newaxis]
        frame = (1.0 - alpha) * frame + alpha * shifted_brick[:, :, :3]
        
        out_frames.append(frame.astype(np.uint8))

    # ensure the first frame is EXACTLY the first_frame.png so there's no subtle color shift
    out_frames[0] = img.copy()

    # We will write frames to a temp dir and then use ffmpeg to encode
    tmp_dir = '/app/tmp_frames'
    os.makedirs(tmp_dir, exist_ok=True)
    
    for i, f in enumerate(out_frames):
        cv2.imwrite(f"{tmp_dir}/frame_{i:04d}.png", f)

    cmd = [
        "ffmpeg", "-y", "-framerate", "16", "-i", f"{tmp_dir}/frame_%04d.png",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "/app/output/video.mp4"
    ]
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
