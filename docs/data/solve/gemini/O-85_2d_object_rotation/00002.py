import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("Could not read /app/first_frame.png")
        
    diff = np.max(np.abs(img.astype(int) - 255), axis=2)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((diff > 0).astype(np.uint8), connectivity=8)

    base_img = img.copy()
    objects = []
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 1000:
            # Erase object from base image
            base_img[labels == i] = 255
            
            # Isolate object
            obj_img = np.full_like(img, 255)
            obj_img[labels == i] = img[labels == i]
            objects.append({'img': obj_img, 'cx': centroids[i][0], 'cy': centroids[i][1]})

    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    
    N_frames = 17
    total_angle = 163.0
    
    for i in range(N_frames):
        angle = i * total_angle / (N_frames - 1)
        
        frame = base_img.copy()
        
        for obj in objects:
            cx, cy = obj['cx'], obj['cy']
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            rotated_obj = cv2.warpAffine(obj['img'], M, (img.shape[1], img.shape[0]), borderValue=(255,255,255), flags=cv2.INTER_LINEAR)
            frame = np.minimum(frame, rotated_obj)
            
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    solve()
