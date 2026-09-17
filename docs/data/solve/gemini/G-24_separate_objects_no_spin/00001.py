import cv2
import numpy as np
import imageio

def generate_video():
    img = cv2.imread('/app/first_frame.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. Identify left objects
    mask_left = np.zeros(img.shape[:2], dtype=bool)
    # The exact colors in RGB
    for color in [[80, 120, 255], [235, 120, 120], [255, 100, 100], [120, 120, 120]]:
        mask_left |= np.all(img == color, axis=-1)
        
    mask_left_uint8 = (mask_left * 255).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_left_uint8, connectivity=8)
    
    # 2. Find best horizontal offsets for each object
    mask_right = np.all(img == [90, 90, 90], axis=-1).astype(np.uint8)
    border_color = np.all(img == [120, 120, 120], axis=-1).astype(np.uint8)
    
    best_offsets = {}
    for i in range(1, num_labels):
        obj_mask = (labels == i).astype(np.uint8)
        obj_border = obj_mask & border_color
        
        best_score = -1
        best_offset = 0
        for x_offset in range(100, 800):
            shifted = np.roll(obj_border, x_offset, axis=1)
            shifted[:, :x_offset] = 0
            
            score = np.sum((shifted > 0) & (mask_right > 0))
            if score > best_score:
                best_score = score
                best_offset = x_offset
                
        best_offsets[i] = best_offset
        print(f"Object {i} offset: {best_offset}, score: {best_score}")
        
    # 3. Prepare background image
    base_img = img.copy()
    base_img[mask_left_uint8 > 0] = [255, 255, 255]
    
    # 4. Generate frames
    frames = []
    num_frames = 30
    for f in range(num_frames):
        frame = base_img.copy()
        
        for i in range(1, num_labels):
            obj_mask = (labels == i).astype(bool)
            dx = int(round(best_offsets[i] * f / (num_frames - 1)))
            
            # Extract object pixels
            obj_pixels = img[obj_mask]
            
            # Shift the mask
            y, x = np.where(obj_mask)
            shifted_x = x + dx
            
            # Draw on frame (ensure we don't go out of bounds, though it shouldn't)
            valid = shifted_x < img.shape[1]
            frame[y[valid], shifted_x[valid]] = obj_pixels[valid]
            
        frames.append(frame)
        
    # 5. Save video
    # H.264, yuv420p, 1024x1024, 16 fps
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print("Video generated at /app/output/video.mp4")

if __name__ == "__main__":
    import os
    if not os.path.exists('/app/output'):
        os.makedirs('/app/output')
    generate_video()
