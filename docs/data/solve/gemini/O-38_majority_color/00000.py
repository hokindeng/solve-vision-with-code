import cv2
import numpy as np
import os
import imageio

def solve():
    img = cv2.imread('/app/first_frame.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray < 255).astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    color_counts = {}
    objects_by_color = {}
    
    for i in range(1, num_labels):
        obj_mask = (labels == i)
        obj_pixels = img[obj_mask]
        
        # filter out black pixels to find the real color
        non_black_pixels = [tuple(p) for p in obj_pixels if tuple(p) != (0, 0, 0)]
        if len(non_black_pixels) > 0:
            unique_colors, counts = np.unique(non_black_pixels, axis=0, return_counts=True)
            majority_color = tuple(unique_colors[np.argmax(counts)])
        else:
            majority_color = (0, 0, 0)
            
        if majority_color not in color_counts:
            color_counts[majority_color] = 0
            objects_by_color[majority_color] = []
        color_counts[majority_color] += 1
        objects_by_color[majority_color].append(i)

    # find the majority color
    majority_color = max(color_counts, key=color_counts.get)
    print(f"Majority color: {majority_color} with {color_counts[majority_color]} objects.")

    # all labels that need to vanish
    vanish_labels = set()
    for c, objs in objects_by_color.items():
        if c != majority_color:
            vanish_labels.update(objs)
            
    # create vanishing mask
    vanish_mask = np.isin(labels, list(vanish_labels))
    
    os.makedirs('/app/output', exist_ok=True)
    frames = []
    
    num_frames = 40
    # ensure it takes 40 frames
    for i in range(num_frames):
        alpha = 1.0 - (i / (num_frames - 1))  # 1.0 down to 0.0
        
        # apply fading to vanish_mask regions
        frame = img.copy().astype(np.float32)
        
        # target is white
        target = np.full_like(frame, 255)
        
        # blend
        frame[vanish_mask] = frame[vanish_mask] * alpha + target[vanish_mask] * (1 - alpha)
        
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        # convert to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, macro_block_size=None, codec='libx264', pixelformat='yuv420p')
    print("Video written successfully.")

if __name__ == '__main__':
    solve()
