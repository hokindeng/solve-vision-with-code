import cv2
import numpy as np
import imageio

def smoothstep(x):
    return x * x * (3 - 2 * x)

def solve():
    img = cv2.imread('/app/first_frame.png')

    def get_symbol(x):
        cell = img[465:560, x:x+95].copy()
        mask = (cell != 255).any(axis=2)
        return cell, mask

    heart, heart_mask = get_symbol(45)
    sym2, sym2_mask = get_symbol(255)
    sym3, sym3_mask = get_symbol(360)
    sym4, sym4_mask = get_symbol(465)

    bg = img.copy()
    for x in [255, 360, 465]:
        bg[465:560, x:x+95] = 255

    frames = []
    num_frames = 54

    def paste(frame, sym, mask, x, y, alpha=1.0):
        roi = frame[y:y+95, x:x+95]
        if alpha == 1.0:
            roi[mask] = sym[mask]
        else:
            roi_float = roi[mask].astype(float)
            sym_float = sym[mask].astype(float)
            blended = roi_float * (1 - alpha) + sym_float * alpha
            roi[mask] = blended.astype(np.uint8)

    for f in range(num_frames):
        frame = bg.copy()
        
        p_slide = np.clip((f - 6) / 24.0, 0.0, 1.0)
        p_slide = smoothstep(p_slide)
        
        p_fade = np.clip((f - 30) / 15.0, 0.0, 1.0)
        
        shift_x = int(p_slide * 105)
        
        paste(frame, sym2, sym2_mask, 255 + shift_x, 465)
        paste(frame, sym3, sym3_mask, 360 + shift_x, 465)
        paste(frame, sym4, sym4_mask, 465 + shift_x, 465)
        
        if p_fade > 0:
            paste(frame, heart, heart_mask, 255, 465, alpha=p_fade)
            paste(frame, heart, heart_mask, 675, 465, alpha=p_fade)
            
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, macro_block_size=None, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    solve()
