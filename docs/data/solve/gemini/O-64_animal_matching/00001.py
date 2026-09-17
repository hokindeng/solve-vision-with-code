import cv2
import numpy as np
import imageio
import os

def get_filled_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled

def main():
    image_path = '/app/first_frame.png'
    output_path = '/app/output/video.mp4'
    os.makedirs('/app/output', exist_ok=True)
    
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read {image_path}")
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    bg_color = tuple(image_rgb[0, 0])
    mask = np.any(image_rgb != bg_color, axis=-1).astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)

    left_comps = []
    right_comps = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if w <= 2 and h > 900:
            continue # separator line
        if centroids[i][0] < 512:
            left_comps.append(i)
        else:
            right_comps.append(i)

    # Match left to right
    matches = {}
    offsets = {}
    for l in left_comps:
        mask_l = (labels == l).astype(np.uint8) * 255
        filled_l = get_filled_mask(mask_l)
        lx, ly, lw, lh = cv2.boundingRect(filled_l)
        template = filled_l[ly:ly+lh, lx:lx+lw]
        
        best_r = -1
        best_val = -1
        best_dx, best_dy = 0, 0
        
        for r in right_comps:
            mask_r = (labels == r).astype(np.uint8) * 255
            filled_r = get_filled_mask(mask_r)
            
            res = cv2.matchTemplate(filled_r, template, cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            if max_val > best_val:
                best_val = max_val
                best_r = r
                best_dx = max_loc[0] - lx
                best_dy = max_loc[1] - ly
                
        matches[l] = best_r
        offsets[l] = (best_dx, best_dy)

    # Prepare static background
    static_bg = image_rgb.copy()
    for l in left_comps:
        static_bg[labels == l] = bg_color

    frames = []
    num_frames = 64
    
    for f in range(num_frames):
        frame = static_bg.copy()
        
        for l in left_comps:
            best_dx, best_dy = offsets[l]
            # Linear interpolation
            f_dx = int(round(best_dx * f / (num_frames - 1)))
            f_dy = int(round(best_dy * f / (num_frames - 1)))
            
            y_idx, x_idx = np.where(labels == l)
            
            y_new = y_idx + f_dy
            x_new = x_idx + f_dx
            
            # Clip just in case, though logically they are within bounds
            valid = (y_new >= 0) & (y_new < frame.shape[0]) & (x_new >= 0) & (x_new < frame.shape[1])
            
            frame[y_new[valid], x_new[valid]] = image_rgb[y_idx[valid], x_idx[valid]]
            
        frames.append(frame)

    # Write video
    # H.264, yuv420p, 1024x1024, 16 fps
    writer = imageio.get_writer(output_path, fps=16, codec='libx264', macro_block_size=None, pixelformat='yuv420p')
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == '__main__':
    main()
