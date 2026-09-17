import cv2
import numpy as np
import imageio
import os

def ease(t):
    if t < 0: return 0
    if t > 1: return 1
    return t * t * (3 - 2 * t)

def solve():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("Could not read /app/first_frame.png")

    # Extract shapes from the first 5 slots
    shapes = []
    for i in range(5):
        x_offset = 44 + 105 * i
        y_offset = 464
        inner = img[y_offset+1:y_offset+96, x_offset+1:x_offset+96].copy()
        mask = np.any(inner != [255, 255, 255], axis=-1)
        shapes.append({'inner': inner, 'mask': mask})

    # Prepare background by clearing inner regions of all slots
    bg = img.copy()
    for i in range(9):
        x0 = 44 + 105 * i + 1
        y0 = 464 + 1
        bg[y0:y0+95, x0:x0+95] = [255, 255, 255]

    def draw(f, inner, mask, x, y):
        x, y = int(round(x)), int(round(y))
        f[y:y+95, x:x+95][mask] = inner[mask]

    frames = []
    for frame_idx in range(54):
        f = bg.copy()
        
        # 1. Slide original shapes (shapes 0..4) to the right by 2 slots
        t_slide = (frame_idx - 5) / 20.0
        e_slide = ease(t_slide)
        
        for i in range(5):
            start_x = 44 + 105 * i + 1
            end_x = 44 + 105 * (i + 2) + 1
            curr_x = start_x + (end_x - start_x) * e_slide
            curr_y = 464 + 1
            draw(f, shapes[i]['inner'], shapes[i]['mask'], curr_x, curr_y)
            
        # 2. Fly new diamonds from reference panel
        # Start coordinates align perfectly with reference panel diamond
        start_fly_x = 899
        start_fly_y = 30
        
        end_fly1_x = 44 + 105 * 0 + 1
        end_fly1_y = 464 + 1
        
        end_fly2_x = 44 + 105 * 1 + 1
        end_fly2_y = 464 + 1
        
        # Diamond 1 (goes to pos 1 / slot 0)
        if frame_idx >= 30:
            t_fly1 = (frame_idx - 30) / 15.0
            e_fly1 = ease(t_fly1)
            curr_x1 = start_fly_x + (end_fly1_x - start_fly_x) * e_fly1
            curr_y1 = start_fly_y + (end_fly1_y - start_fly_y) * e_fly1
            draw(f, shapes[4]['inner'], shapes[4]['mask'], curr_x1, curr_y1)
            
        # Diamond 2 (goes to pos 2 / slot 1)
        if frame_idx >= 35:
            t_fly2 = (frame_idx - 35) / 15.0
            e_fly2 = ease(t_fly2)
            curr_x2 = start_fly_x + (end_fly2_x - start_fly_x) * e_fly2
            curr_y2 = start_fly_y + (end_fly2_y - start_fly_y) * e_fly2
            draw(f, shapes[4]['inner'], shapes[4]['mask'], curr_x2, curr_y2)
            
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))

    # Save video with required parameters
    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite(
        '/app/output/video.mp4',
        frames,
        fps=16,
        codec='libx264',
        pixelformat='yuv420p',
        macro_block_size=None
    )

if __name__ == '__main__':
    solve()
