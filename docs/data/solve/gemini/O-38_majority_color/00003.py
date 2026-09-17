import cv2
import numpy as np
import imageio
import os

def main():
    # Read the input image
    img = cv2.imread('/app/first_frame.png')
    
    # Identify unique colors
    colors = np.unique(img.reshape(-1, 3), axis=0)
    bg_color = np.array([255, 255, 255])
    outline_color = np.array([0, 0, 0])
    
    # Extract only object colors
    object_colors = [c for c in colors if not np.array_equal(c, bg_color) and not np.array_equal(c, outline_color)]
    masks = {tuple(c): cv2.inRange(img, c, c) for c in object_colors}
    
    # Count objects of each color
    color_counts = {}
    for c in object_colors:
        num, _, stats, _ = cv2.connectedComponentsWithStats(masks[tuple(c)], connectivity=8)
        # Exclude background component of the mask (which has label 0)
        color_counts[tuple(c)] = sum(1 for s in stats[1:] if s[cv2.CC_STAT_AREA] > 0)
        
    majority_color_tuple = max(color_counts, key=color_counts.get)
    
    # Use distance transforms to assign each black outline pixel to the nearest object color
    black_mask = cv2.inRange(img, outline_color, outline_color)
    dist_maps = np.stack([cv2.distanceTransform(cv2.bitwise_not(masks[tuple(c)]), cv2.DIST_L2, 3) for c in object_colors], axis=-1)
    
    # Break ties in favor of the majority color by slightly reducing its distance
    maj_idx = [tuple(c) for c in object_colors].index(majority_color_tuple)
    dist_maps_modified = dist_maps.copy()
    dist_maps_modified[:, :, maj_idx] -= 1e-5
    
    closest_idx = np.argmin(dist_maps_modified, axis=-1)
    
    # Create a mask for pixels that should vanish
    vanish_mask = np.zeros(img.shape[:2], dtype=bool)
    
    # Outline pixels that do not belong to the majority color
    vanish_mask[(black_mask > 0) & (closest_idx != maj_idx)] = True
    
    # Object pixels that are not the majority color
    for i, c in enumerate(object_colors):
        if i != maj_idx:
            vanish_mask |= (masks[tuple(c)] > 0)
            
    # Prepare to generate video
    os.makedirs('/app/output', exist_ok=True)
    out_path = '/app/output/video.mp4'
    writer = imageio.get_writer(out_path, fps=16, codec='libx264', pixelformat='yuv420p')
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    white_rgb = np.array([255, 255, 255], dtype=np.float32)
    
    # Generate 40 frames
    for i in range(40):
        alpha = i / 39.0
        frame = img_rgb.copy().astype(np.float32)
        
        # Vanish: interpolate towards the background color (white)
        frame[vanish_mask] = frame[vanish_mask] * (1 - alpha) + white_rgb * alpha
        
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    main()
