import cv2
import numpy as np
import imageio
import os

def solve():
    img_bgr = cv2.imread('/app/first_frame.png')
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Identify objects by finding non-white components
    mask = np.any(img_rgb != [255, 255, 255], axis=-1).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Sort components by area to find the two largest (objects)
    comps = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        comps.append((area, i))
    
    comps.sort(reverse=True)
    obj1_idx = comps[0][1]
    obj2_idx = comps[1][1]
    
    cx1, cy1 = centroids[obj1_idx]
    cx2, cy2 = centroids[obj2_idx]
    
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    
    num_frames = 17
    target_angle = -21.0 # -21 degrees for clockwise
    
    def rotate_patch(res, cx, cy, angle, size=300):
        x = int(cx - size/2)
        y = int(cy - size/2)
        
        # Extract patch from the original image
        patch = img_rgb[y:y+size, x:x+size]
        
        # Sub-pixel centroid within the patch
        pcx = cx - x
        pcy = cy - y
        
        M = cv2.getRotationMatrix2D((pcx, pcy), angle, 1.0)
        rot_patch = cv2.warpAffine(patch, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
        
        res[y:y+size, x:x+size] = rot_patch

    for i in range(num_frames):
        # Linearly interpolate angle from 0 to target_angle
        angle = target_angle * i / (num_frames - 1)
        
        frame = img_rgb.copy()
        
        # Rotate both objects
        rotate_patch(frame, cx1, cy1, angle, size=300)
        rotate_patch(frame, cx2, cy2, angle, size=300)
        
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    solve()
