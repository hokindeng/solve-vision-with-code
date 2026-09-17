import numpy as np
from PIL import Image
import imageio
import os

def solve():
    img = np.array(Image.open('/app/first_frame.png').convert('RGB'))

    # Extract the two balls from the first frame
    ball1 = img[532:773, 203:444].copy().astype(np.uint16)
    ball2 = img[471:712, 615:856].copy().astype(np.uint16)

    m1 = np.any(ball1 != 255, axis=-1)
    m2 = np.any(ball2 != 255, axis=-1)

    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)

    for i in range(80):
        t = i / 79.0
        
        frame = np.ones((1024, 1024, 3), dtype=np.uint16) * 255
        
        # Centers with exact midpoint targeting
        x1 = int(round(323 + t * (529 - 323)))
        y1 = int(round(652 + t * (621.5 - 652)))
        x2 = int(round(735 + t * (529 - 735)))
        y2 = int(round(591 + t * (621.5 - 591)))
        
        # Bounding boxes
        x1_min, x1_max = x1 - 120, x1 + 121
        y1_min, y1_max = y1 - 120, y1 + 121
        x2_min, x2_max = x2 - 120, x2 + 121
        y2_min, y2_max = y2 - 120, y2 + 121
        
        F1_color = np.zeros((1024, 1024, 3), dtype=np.uint16)
        F1_mask = np.zeros((1024, 1024), dtype=bool)
        
        F2_color = np.zeros((1024, 1024, 3), dtype=np.uint16)
        F2_mask = np.zeros((1024, 1024), dtype=bool)
        
        F1_color[y1_min:y1_max, x1_min:x1_max] = ball1
        F1_mask[y1_min:y1_max, x1_min:x1_max] = m1
        
        F2_color[y2_min:y2_max, x2_min:x2_max] = ball2
        F2_mask[y2_min:y2_max, x2_min:x2_max] = m2
        
        both = F1_mask & F2_mask
        only1 = F1_mask & ~F2_mask
        only2 = F2_mask & ~F1_mask
        
        frame[both] = np.minimum(F1_color[both] + F2_color[both], 255)
        frame[only1] = F1_color[only1]
        frame[only2] = F2_color[only2]
        
        writer.append_data(frame.astype(np.uint8))
        
    writer.close()

if __name__ == '__main__':
    solve()
