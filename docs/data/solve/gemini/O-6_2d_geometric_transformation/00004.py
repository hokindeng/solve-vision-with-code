import cv2
import numpy as np
import imageio
import os

def solve():
    img_path = '/app/first_frame.png'
    img = cv2.imread(img_path)
    
    # Define colors in BGR
    bg_color = np.array([240, 240, 240])
    target_color = np.array([100, 100, 100])
    poly_color = np.array([174, 152, 132])
    outline_color = np.array([50, 50, 50])
    marker_color1 = np.array([0, 0, 0])
    marker_color2 = np.array([255, 255, 255])
    
    mask_target = cv2.inRange(img, target_color, target_color)
    mask_poly = cv2.inRange(img, poly_color, poly_color)
    mask_outline = cv2.inRange(img, outline_color, outline_color)
    mask_moving = cv2.bitwise_or(mask_poly, mask_outline)
    
    mask_marker1 = cv2.inRange(img, marker_color1, marker_color1)
    mask_marker2 = cv2.inRange(img, marker_color2, marker_color2)
    mask_marker = cv2.bitwise_or(mask_marker1, mask_marker2)
    
    # Find center
    coords = cv2.findNonZero(mask_marker).reshape(-1, 2)
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    
    # Find best angle
    def find_best(start, end, step):
        best_a = start
        best_s = -1
        for a in np.arange(start, end + step/2, step):
            M = cv2.getRotationMatrix2D((cx, cy), a, 1.0)
            rotated = cv2.warpAffine(mask_moving, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)
            s = cv2.countNonZero(cv2.bitwise_and(rotated, mask_target))
            if s > best_s:
                best_s = s
                best_a = a
        return best_a, best_s

    print("Finding exact target angle...")
    a1, s1 = find_best(0, 359, 1)
    a2, s2 = find_best(a1 - 1.0, a1 + 1.0, 0.1)
    target_angle, _ = find_best(a2 - 0.1, a2 + 0.1, 0.01)
    
    print(f"Target angle determined: {target_angle:.3f}")
    
    clean_bg = img.copy()
    clean_bg[mask_moving > 0] = bg_color
    
    moving_bgr = np.zeros_like(img)
    moving_bgr[mask_moving > 0] = img[mask_moving > 0]
    
    marker_bgr = np.zeros_like(img)
    marker_bgr[mask_marker > 0] = img[mask_marker > 0]
    
    frames = []
    num_frames = 70
    angles = np.linspace(0, target_angle, num_frames)
    
    print("Generating frames...")
    for a in angles:
        M = cv2.getRotationMatrix2D((cx, cy), a, 1.0)
        rot_mask = cv2.warpAffine(mask_moving, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)
        rot_moving = cv2.warpAffine(moving_bgr, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)
        
        frame = clean_bg.copy()
        frame[rot_mask > 0] = rot_moving[rot_mask > 0]
        
        # Redraw marker on top to ensure it remains unchanged
        frame[mask_marker > 0] = marker_bgr[mask_marker > 0]
        
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    out_dir = '/app/output'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'video.mp4')
    
    print(f"Saving video to {out_path}...")
    imageio.mimwrite(out_path, frames, fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=16)
    print("Done!")

if __name__ == '__main__':
    solve()
