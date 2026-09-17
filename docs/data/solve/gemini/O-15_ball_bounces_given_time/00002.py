import cv2
import numpy as np
import imageio
import os

def get_pos(d):
    if d <= 179:
        return (607 - d, 272 - d)
    d -= 179
    if d <= 335:
        return (428 - d, 93 + d)
    d -= 335
    if d <= 503:
        return (93 + d, 428 + d)
    return (596, 931)

def main():
    img = cv2.imread('/app/first_frame.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    ball_color = np.array([255, 192, 203], dtype=np.uint8)
    arrow_color = np.array([255, 140, 0], dtype=np.uint8)
    wall_color = np.array([100, 100, 100], dtype=np.uint8)
    bg_color = np.array([255, 255, 255], dtype=np.uint8)
    
    # Original ball mask
    c2 = (img == ball_color).all(axis=-1)
    
    # Bounding box of the original ball
    bbox = c2[242:303, 577:638]
    
    # Reconstruct full ball by mirroring
    full_ball = bbox.copy()
    full_ball = full_ball | np.flip(full_ball, axis=0) | np.flip(full_ball, axis=1) | np.flip(np.flip(full_ball, axis=0), axis=1)
    
    # Get relative coordinates of full ball
    y_coords, x_coords = np.where(full_ball)
    # Center of the 61x61 bounding box is (30, 30)
    dx = x_coords - 30
    dy = y_coords - 30
    
    # Base image with original ball removed
    base_img = img.copy()
    base_img[c2] = bg_color
    
    # Foreground mask (arrow + walls)
    fg_mask = (img == arrow_color).all(axis=-1) | (img == wall_color).all(axis=-1)
    
    frames = []
    for i in range(80):
        d = i * 1017 / 79
        cx, cy = get_pos(d)
        cx = int(round(cx))
        cy = int(round(cy))
        
        frame = base_img.copy()
        frame_y = cy + dy
        frame_x = cx + dx
        
        # Clip to valid range just in case
        valid = (frame_y >= 0) & (frame_y < img.shape[0]) & (frame_x >= 0) & (frame_x < img.shape[1])
        frame[frame_y[valid], frame_x[valid]] = ball_color
        
        # Restore foreground elements (arrow and walls)
        frame[fg_mask] = img[fg_mask]
        
        frames.append(frame)
    
    os.makedirs('/app/output', exist_ok=True)
    out_path = '/app/output/video.mp4'
    
    writer = imageio.get_writer(out_path, fps=16, codec='libx264', macro_block_size=None, pixelformat='yuv420p')
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == '__main__':
    main()
