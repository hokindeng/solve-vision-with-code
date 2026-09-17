import cv2
import numpy as np
import imageio
import os

def create_video():
    img = cv2.imread('/app/first_frame.png')
    
    # Identify objects in the left half of the image
    left_mask = (img[:, :500] != 255).any(axis=2).astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(left_mask, connectivity=8)

    full_labels = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)
    full_labels[:, :500] = labels

    # Known required horizontal shifts determined analytically
    shifts_by_y = {
        327: 638,
        507: 432,
        552: 451
    }
    
    objects = []
    
    # Create clean background by masking out initial object positions
    bg = img.copy()
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        obj_mask = (full_labels == i)
        bg[obj_mask] = [255, 255, 255]
        
        # Match each object to its corresponding shift based on its y-coordinate
        closest_y = min(shifts_by_y.keys(), key=lambda k: abs(k - y))
        final_dx = shifts_by_y[closest_y]
        
        objects.append({
            'mask': obj_mask,
            'dx_total': final_dx,
            'x': x, 'y': y, 'w': w, 'h': h
        })
        
    num_frames = 30
    fps = 16
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, macro_block_size=None, codec='libx264', pixelformat='yuv420p')
    
    for frame_idx in range(num_frames):
        # Progress variable t goes from 0.0 to 1.0
        t = frame_idx / (num_frames - 1)
        
        # Start with the clean background for this frame
        frame = bg.copy()
        
        # Draw each object at its interpolated position
        for obj in objects:
            dx = int(round(obj['dx_total'] * t))
            
            ys, xs = np.where(obj['mask'])
            shifted_xs = xs + dx
            
            # Copy object pixels to new position
            frame[ys, shifted_xs] = img[ys, xs]
            
        # Write frame to video
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    os.makedirs('/app/output', exist_ok=True)
    create_video()
