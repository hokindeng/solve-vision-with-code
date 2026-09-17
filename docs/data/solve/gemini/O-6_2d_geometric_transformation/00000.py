import cv2
import numpy as np
import math
import imageio

def main():
    img = cv2.imread('/app/first_frame.png')
    
    cx, cy = 434, 434
    
    # Identify masks
    mask_poly = (cv2.inRange(img, np.array([77,129,50]), np.array([77,129,50]))) > 0
    mask_line = (cv2.inRange(img, np.array([50,50,50]), np.array([50,50,50]))) > 0
    mask_moving = mask_poly | mask_line
    
    # Identify center marker (to redraw on top)
    mask_center1 = (cv2.inRange(img, np.array([0,0,0]), np.array([0,0,0]))) > 0
    mask_center2 = (cv2.inRange(img, np.array([255,255,255]), np.array([255,255,255]))) > 0
    mask_center = mask_center1 | mask_center2
    
    # Extract center marker pixels
    center_img = np.zeros_like(img)
    center_img[mask_center] = img[mask_center]
    
    # Create background by filling moving parts with background color
    bg = img.copy()
    bg[mask_moving] = [240, 240, 240]
    
    # Create RGBA image of moving parts
    moving_rgba = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
    moving_rgba[mask_poly] = [77, 129, 50, 255]
    moving_rgba[mask_line] = [50, 50, 50, 255]
    
    # Total angle to rotate
    # Solid line angle
    y_line, x_line = np.where(mask_line)
    dists = np.sqrt((x_line - cx)**2 + (y_line - cy)**2)
    max_idx = np.argmax(dists)
    lx, ly = x_line[max_idx], y_line[max_idx]
    
    # Target line angle
    mask_target = (cv2.inRange(img, np.array([100,100,100]), np.array([100,100,100]))) > 0
    y_tgt, x_tgt = np.where(mask_target)
    dists_tgt = np.sqrt((x_tgt - cx)**2 + (y_tgt - cy)**2)
    max_idx_tgt = np.argmax(dists_tgt)
    tx, ty = x_tgt[max_idx_tgt], y_tgt[max_idx_tgt]
    
    angle_solid = math.degrees(math.atan2(ly - cy, lx - cx))
    angle_target = math.degrees(math.atan2(ty - cy, tx - cx))
    
    # CCW visual rotation angle
    # In cv2, positive angle is CCW visually.
    diff = (angle_solid - angle_target) % 360
    total_angle = diff
    
    num_frames = 70
    fps = 16
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, codec='libx264', pixelformat='yuv420p')
    
    for i in range(num_frames):
        # Linearly interpolate angle
        angle = (i / (num_frames - 1)) * total_angle
        
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated_moving = cv2.warpAffine(moving_rgba, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)
        
        # Blend
        alpha = (rotated_moving[:, :, 3] > 0)
        
        frame = bg.copy()
        frame[alpha] = rotated_moving[alpha, :3]
        
        # Draw center marker on top to be perfectly safe
        frame[mask_center] = center_img[mask_center]
        
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    main()
