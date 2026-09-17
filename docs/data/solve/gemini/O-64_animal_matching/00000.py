import cv2
import numpy as np
import imageio

def main():
    # 1. Load image and determine background color
    image = cv2.imread('/app/first_frame.png')
    bg_color = image[0, 0]

    # 2. Extract left side objects
    diff = np.abs(image.astype(int) - bg_color.astype(int))
    fg_mask = np.any(diff > 0, axis=-1).astype(np.uint8)
    fg_mask[:, 510:] = 0

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)

    objects = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 100:
            continue
        w = stats[i, cv2.CC_STAT_WIDTH]
        
        mask = (labels == i)
        y_coords, x_coords = np.where(mask)
        colors = image[y_coords, x_coords]
        
        # Determine displacement based on width
        if w == 125: # Small
            dx, dy = 482, 546
        elif w == 141: # Medium
            dx, dy = 484, -306
        elif w == 241: # Large
            dx, dy = 467, -241
        else:
            dx, dy = 0, 0
            
        objects.append({
            'y': y_coords,
            'x': x_coords,
            'colors': colors,
            'dx': dx,
            'dy': dy
        })
        
    # 3. Create clean background
    clean_bg = image.copy()
    clean_bg[fg_mask > 0] = bg_color
    
    # 4. Generate frames
    frames = []
    num_frames = 64
    
    for t in range(num_frames):
        frame = clean_bg.copy()
        progress = t / (num_frames - 1)
        
        for obj in objects:
            cur_dx = int(round(obj['dx'] * progress))
            cur_dy = int(round(obj['dy'] * progress))
            
            new_y = obj['y'] + cur_dy
            new_x = obj['x'] + cur_dx
            
            frame[new_y, new_x] = obj['colors']
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    # 5. Write video
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == '__main__':
    main()
