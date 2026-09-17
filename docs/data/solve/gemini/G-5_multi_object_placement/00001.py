import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    bg_color = np.array([255, 255, 255], dtype=np.uint8)
    
    unique_colors = np.unique(img.reshape(-1, 3), axis=0)
    bg_color_list = [255, 255, 255]
    
    objects = []
    stars = []
    
    bg_img = img.copy()
    
    for color in unique_colors:
        if list(color) == bg_color_list:
            continue
            
        mask = cv2.inRange(img, color, color)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
        
        comps = []
        for j in range(1, num_labels):
            area = stats[j, cv2.CC_STAT_AREA]
            cx, cy = centroids[j]
            comps.append({
                'label': j,
                'area': area,
                'centroid': (cx, cy),
                'color': color,
                'mask': (labels == j)
            })
            
        comps.sort(key=lambda x: x['area'])
        star = comps[0]
        obj = comps[-1]
        
        stars.append(star)
        objects.append(obj)
        
        bg_img[obj['mask']] = bg_color

    frames = []
    num_frames = 48
    
    os.makedirs('/app/output', exist_ok=True)
    
    for f in range(num_frames):
        t = f / (num_frames - 1)
        frame = bg_img.copy()
        
        for obj, star in zip(objects, stars):
            start_x, start_y = obj['centroid']
            end_x, end_y = star['centroid']
            
            curr_x = start_x + (end_x - start_x) * t
            curr_y = start_y + (end_y - start_y) * t
            
            dx = int(round(curr_x - start_x))
            dy = int(round(curr_y - start_y))
            
            ys, xs = np.where(obj['mask'])
            new_ys = ys + dy
            new_xs = xs + dx
            
            valid = (new_ys >= 0) & (new_ys < frame.shape[0]) & (new_xs >= 0) & (new_xs < frame.shape[1])
            new_ys = new_ys[valid]
            new_xs = new_xs[valid]
            
            frame[new_ys, new_xs] = obj['color']
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, macro_block_size=None, codec='libx264', format='FFMPEG', pixelformat='yuv420p')

if __name__ == '__main__':
    solve()
