import numpy as np
import imageio
from PIL import Image
import os

def main():
    img = Image.open('/app/first_frame.png')
    img_np = np.array(img)
    h, w, c = img_np.shape

    bg_color = np.array([255, 255, 255], dtype=np.uint8)
    c1_color = np.array([230, 25, 75], dtype=np.uint8)
    c2_color = np.array([250, 190, 212], dtype=np.uint8)

    c1_mask = np.all(img_np == c1_color, axis=-1)
    c2_mask = np.all(img_np == c2_color, axis=-1)

    cx1, cy1 = 315, 307
    cx2, cy2 = 835, 716

    target_cx, target_cy = 512, 512

    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)

    c1_y, c1_x = np.where(c1_mask)
    c2_y, c2_x = np.where(c2_mask)

    for i in range(40):
        alpha = i / 39.0
        
        cur_cx1 = int(round(cx1 + (target_cx - cx1) * alpha))
        cur_cy1 = int(round(cy1 + (target_cy - cy1) * alpha))
        
        cur_cx2 = int(round(cx2 + (target_cx - cx2) * alpha))
        cur_cy2 = int(round(cy2 + (target_cy - cy2) * alpha))
        
        dx1 = cur_cx1 - cx1
        dy1 = cur_cy1 - cy1
        
        dx2 = cur_cx2 - cx2
        dy2 = cur_cy2 - cy2
        
        frame = np.copy(img_np)
        
        # We need to erase the original circles first
        frame[c1_mask] = bg_color
        frame[c2_mask] = bg_color
        
        # Then draw them in their new positions
        new_c1_x = c1_x + dx1
        new_c1_y = c1_y + dy1
        
        new_c2_x = c2_x + dx2
        new_c2_y = c2_y + dy2
        
        # To handle any overlap and keep z-order: larger first, smaller second.
        # c1 is larger (radius 301), c2 is smaller (radius 175)
        frame[new_c1_y, new_c1_x] = c1_color
        frame[new_c2_y, new_c2_x] = c2_color
        
        writer.append_data(frame)

    writer.close()

if __name__ == '__main__':
    main()
