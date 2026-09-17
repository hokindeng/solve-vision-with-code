import cv2
import numpy as np
import imageio
import os

def generate_video():
    # Read the initial frame
    img = cv2.imread('/app/first_frame.png')
    bg_color = np.array([220, 220, 220])

    # Bounding box and patch for Object 1
    # Bounding box found exactly aligns with the object
    x1, y1, w1, h1 = 416, 177, 137, 137
    obj1_patch = img[y1:y1+h1, x1:x1+w1].copy()
    mask1 = np.any(np.abs(obj1_patch.astype(int) - bg_color) > 0, axis=2)

    # Bounding box and patch for Object 2
    x2, y2, w2, h2 = 432, 531, 137, 137
    obj2_patch = img[y2:y2+h2, x2:x2+w2].copy()
    mask2 = np.any(np.abs(obj2_patch.astype(int) - bg_color) > 0, axis=2)

    # Create clean background by removing both objects from the initial frame
    clean_bg = img.copy()
    clean_bg[y1:y1+h1, x1:x1+w1][mask1] = bg_color
    clean_bg[y2:y2+h2, x2:x2+w2][mask2] = bg_color

    frames = []
    num_frames = 35

    for i in range(num_frames):
        # Time variable between 0 and 1
        t = i / (num_frames - 1)
        
        # Calculate intermediate offsets
        # Total displacement for Obj 1: (58, 175) to fit exactly inside Target 1
        dx1 = int(round(58 * t))
        dy1 = int(round(175 * t))
        
        # Total displacement for Obj 2: (-76, 315) to fit exactly inside Target 2
        dx2 = int(round(-76 * t))
        dy2 = int(round(315 * t))
        
        frame = clean_bg.copy()
        
        # Paste Object 1 at interpolated position
        nx1, ny1 = x1 + dx1, y1 + dy1
        frame[ny1:ny1+h1, nx1:nx1+w1][mask1] = obj1_patch[mask1]
        
        # Paste Object 2 at interpolated position
        nx2, ny2 = x2 + dx2, y2 + dy2
        frame[ny2:ny2+h2, nx2:nx2+w2][mask2] = obj2_patch[mask2]
        
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    # Save the output video
    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    generate_video()
