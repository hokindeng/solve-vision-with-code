import cv2
import numpy as np
import imageio
import os

def main():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("Could not read /app/first_frame.png")
        
    pixels = img.reshape(-1, 3)
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    bg_color = unique_colors[np.argmax(counts)]
    
    clean_bg = img.copy()
    objects = []
    
    for c in unique_colors:
        if np.array_equal(c, bg_color):
            continue
            
        mask = cv2.inRange(img, c, c)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        star_info = None
        obj_info = None
        
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            comp_mask = (labels == i).astype(np.uint8) * 255
            
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0: continue
            cnt = contours[0]
            c_area = cv2.contourArea(cnt)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            
            solidity = c_area / hull_area if hull_area > 0 else 0
            
            cx = x + w / 2.0
            cy = y + h / 2.0
            
            if solidity < 0.5:
                star_info = {'cx': cx, 'cy': cy}
            else:
                obj_info = {'mask': comp_mask, 'cx': cx, 'cy': cy}
                # Remove object from background by filling its mask with background color
                clean_bg[comp_mask > 0] = bg_color
                
        if star_info is not None and obj_info is not None:
            objects.append({
                'color': c,
                'mask': obj_info['mask'],
                'cx': obj_info['cx'],
                'cy': obj_info['cy'],
                'tx': star_info['cx'],
                'ty': star_info['cy']
            })

    os.makedirs('/app/output', exist_ok=True)
    frames = []
    N_FRAMES = 48
    
    for f in range(N_FRAMES):
        progress = f / (N_FRAMES - 1)
        frame = clean_bg.copy()
        
        for obj in objects:
            current_cx = obj['cx'] + (obj['tx'] - obj['cx']) * progress
            current_cy = obj['cy'] + (obj['ty'] - obj['cy']) * progress
            
            dx = current_cx - obj['cx']
            dy = current_cy - obj['cy']
            
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            warped_mask = cv2.warpAffine(obj['mask'], M, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            frame[warped_mask > 0] = obj['color']
            
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', pixelformat='yuv420p', ffmpeg_params=['-crf', '17'])

if __name__ == '__main__':
    main()
