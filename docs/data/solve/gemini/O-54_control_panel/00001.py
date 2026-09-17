import numpy as np
from PIL import Image
import imageio
import os

def solve():
    img_path = '/app/first_frame.png'
    out_path = '/app/output/video.mp4'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    img = Image.open(img_path).convert('RGB')
    arr = np.array(img)
    
    # Precompute mask for Purple pixels (the indicator lights on units 1 and 3)
    # Purple in RGB is [128, 0, 128]
    mask_purple = np.all(arr == [128, 0, 128], axis=-1)
    
    writer = imageio.get_writer(out_path, fps=16, codec='libx264', pixelformat='yuv420p')
    
    def draw_mark(frame, tx, mx):
        # Draw the small white plus/circle marks on the tracks
        frame[620+54:620+59, tx+mx-2:tx+mx+3] = [240, 240, 240]
        frame[620+54, tx+mx-2] = [0, 0, 0]
        frame[620+54, tx+mx+2] = [0, 0, 0]
        frame[620+58, tx+mx-2] = [0, 0, 0]
        frame[620+58, tx+mx+2] = [0, 0, 0]

    def draw_lever(frame, lx):
        # Draw the grey block representing the lever
        frame[648:648+57, lx:lx+57] = [128, 128, 128]

    # Generate 24 frames for 1.5 seconds at 16fps
    for f in range(24):
        p = f / 23.0
        frame = arr.copy()
        
        # 1. Update Lights smoothly based on lever position progress
        # Purple [128, 0, 128] -> Blue [0, 0, 255]
        r = int(round(128 * (1 - p)))
        g = 0
        b = int(round(128 * (1 - p) + 255 * p))
        frame[mask_purple] = [r, g, b]
        
        # 2. Update Levers: Slide Unit 1 and Unit 3 levers to the middle
        l1_x = int(round(253 - 77 * p))
        l3_x = int(round(867 - 77 * p))
        
        # Redraw Tracks and Levers for all 3 units
        for tx in [89, 396, 703]:
            # Clear black track
            frame[620:734, tx:tx+232] = [0, 0, 0]
            # Draw the 3 indicator marks on the track
            for mx in [33, 115, 197]:
                draw_mark(frame, tx, mx)
                
        # Draw the levers at their calculated positions
        draw_lever(frame, l1_x)  # Unit 1
        draw_lever(frame, 483)   # Unit 2 (always in the middle)
        draw_lever(frame, l3_x)  # Unit 3
        
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    solve()
