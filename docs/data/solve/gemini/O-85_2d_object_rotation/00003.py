import cv2
import numpy as np
import imageio
import os

def solve():
    img_path = '/app/first_frame.png'
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read {img_path}")
        
    H, W = img.shape[:2]
    
    # Find background color (most common color)
    colors, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
    bg_color = colors[np.argmax(counts)]
    
    # Create mask of foreground
    mask = np.any(img != bg_color, axis=-1).astype(np.uint8)
    
    # Find the largest connected component in the foreground
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    if num_labels <= 1:
        raise ValueError("No object found in the image.")
        
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    M = (labels == largest_label)
    
    # Compute precise centroid
    moments = cv2.moments(M.astype(np.uint8))
    if moments["m00"] != 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
    else:
        cx, cy = centroids[largest_label]
        
    # Prepare object RGBA
    obj_rgba = np.zeros((H, W, 4), dtype=np.uint8)
    obj_rgba[..., :3] = bg_color
    obj_rgba[M, :3] = img[M]
    obj_rgba[M, 3] = 255
    
    # Prepare base image
    base_img = img.copy()
    base_img[M] = bg_color
    
    out_dir = '/app/output'
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, 'video.mp4')
    
    writer = imageio.get_writer(out_path, fps=16, codec='libx264', format='FFMPEG', pixelformat='yuv420p', macro_block_size=None)
    
    num_frames = 17
    bv = (int(bg_color[0]), int(bg_color[1]), int(bg_color[2]), 0)
    for i in range(num_frames):
        angle = -180.0 * i / (num_frames - 1)
        
        if i == 0:
            frame = img.copy()
        else:
            matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            rotated_obj = cv2.warpAffine(obj_rgba, matrix, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=bv)
            
            alpha = rotated_obj[..., 3:] / 255.0
            frame = (rotated_obj[..., :3] * alpha + base_img * (1 - alpha)).astype(np.uint8)
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    solve()
