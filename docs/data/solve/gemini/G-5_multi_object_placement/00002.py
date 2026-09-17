import cv2
import numpy as np
import imageio
import os
from collections import defaultdict

def solve():
    img = cv2.imread('/app/first_frame.png')
    bg_color = np.array([255, 255, 255])
    
    # Create mask for all non-background pixels
    mask = np.any(img != bg_color, axis=-1).astype(np.uint8)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Group components by color
    color_groups = defaultdict(list)
    for i in range(1, num_labels):
        comp_mask = (labels == i)
        # get color of this component (first pixel is enough since it's solid color)
        color = tuple(int(x) for x in img[comp_mask][0])
        area = stats[i, cv2.CC_STAT_AREA]
        color_groups[color].append({'id': i, 'area': area, 'stats': stats[i]})
        
    objects = []
    stars = []
    
    # For each color, the smaller component is the star, the larger is the object
    for color, comps in color_groups.items():
        if len(comps) == 2:
            comps.sort(key=lambda x: x['area'])
            stars.append(comps[0])
            objects.append(comps[1])
            
    # Prepare clean background (remove objects)
    clean_bg = img.copy()
    for obj in objects:
        clean_bg[labels == obj['id']] = bg_color
        
    # Calculate translations
    translations = {}
    for obj in objects:
        # Find matching star by color
        obj_color = tuple(int(x) for x in img[labels == obj['id']][0])
        matching_star = next(s for s in stars if tuple(int(x) for x in img[labels == s['id']][0]) == obj_color)
        
        o_stat = obj['stats']
        s_stat = matching_star['stats']
        
        ocx = o_stat[cv2.CC_STAT_LEFT] + o_stat[cv2.CC_STAT_WIDTH] / 2.0
        ocy = o_stat[cv2.CC_STAT_TOP] + o_stat[cv2.CC_STAT_HEIGHT] / 2.0
        
        scx = s_stat[cv2.CC_STAT_LEFT] + s_stat[cv2.CC_STAT_WIDTH] / 2.0
        scy = s_stat[cv2.CC_STAT_TOP] + s_stat[cv2.CC_STAT_HEIGHT] / 2.0
        
        translations[obj['id']] = (scx - ocx, scy - ocy)

    num_frames = 48
    frames = []

    for f in range(num_frames):
        progress = f / (num_frames - 1)
        frame = clean_bg.copy()
        
        for obj in objects:
            dx, dy = translations[obj['id']]
            tx, ty = int(round(dx * progress)), int(round(dy * progress))
            
            obj_mask = (labels == obj['id'])
            ys, xs = np.where(obj_mask)
            
            new_ys = ys + ty
            new_xs = xs + tx
            
            valid = (new_ys >= 0) & (new_ys < img.shape[0]) & (new_xs >= 0) & (new_xs < img.shape[1])
            frame[new_ys[valid], new_xs[valid]] = img[ys[valid], xs[valid]]
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, macro_block_size=None, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    solve()
