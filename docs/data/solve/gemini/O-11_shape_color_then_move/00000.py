import numpy as np
from PIL import Image
import cv2
import os
import imageio

def solve():
    arr = np.array(Image.open('/app/first_frame.png'))
    bg = np.array([255, 255, 255], dtype=np.uint8)

    D_slice = arr[626:739, 83:308].copy()
    teal = np.array([45, 229, 168], dtype=np.float32)
    green = np.array([7, 153, 80], dtype=np.float32)

    mask_D = np.all(D_slice == [45, 229, 168], axis=-1)
    mask_d_border = np.all(D_slice == [0, 0, 0], axis=-1)

    E_y, E_x = 626, 370
    F_y, F_x = 651, 657

    clean_arr = arr.copy()
    clean_arr[650:715, 460:505] = bg
    clean_arr[650:715, 750:790] = bg

    frames = []
    for f in range(60):
        frame = clean_arr.copy()

        # Phase 1: Draw E (pos 2)
        if f <= 7:
            # Fading from ?1 to Teal
            alpha = f / 7.0
            roi_start = arr[E_y:E_y+113, E_x:E_x+225].copy()
            roi_end = clean_arr[E_y:E_y+113, E_x:E_x+225].copy()
            roi_end[mask_D] = teal.astype(np.uint8)
            roi_end[mask_d_border] = [0, 0, 0]
            frame[E_y:E_y+113, E_x:E_x+225] = cv2.addWeighted(roi_start, 1-alpha, roi_end, alpha, 0)
        elif f <= 22:
            # Teal recolors to Green
            alpha = (f - 8) / 14.0
            color = teal * (1 - alpha) + green * alpha
            roi = clean_arr[E_y:E_y+113, E_x:E_x+225].copy()
            roi[mask_D] = color.astype(np.uint8)
            roi[mask_d_border] = [0, 0, 0]
            frame[E_y:E_y+113, E_x:E_x+225] = roi
        else:
            # Solid Green
            roi = clean_arr[E_y:E_y+113, E_x:E_x+225].copy()
            roi[mask_D] = green.astype(np.uint8)
            roi[mask_d_border] = [0, 0, 0]
            frame[E_y:E_y+113, E_x:E_x+225] = roi

        # Phase 2: Draw F (pos 3)
        if f <= 29:
            # Just ?2
            frame[650:715, 750:790] = arr[650:715, 750:790]
        elif f <= 37:
            # Fading from ?2 to Green at E_y
            alpha = (f - 30) / 7.0
            roi_start = arr[E_y:E_y+113, F_x:F_x+225].copy()
            roi_end = clean_arr[E_y:E_y+113, F_x:F_x+225].copy()
            roi_end[mask_D] = green.astype(np.uint8)
            roi_end[mask_d_border] = [0, 0, 0]
            frame[E_y:E_y+113, F_x:F_x+225] = cv2.addWeighted(roi_start, 1-alpha, roi_end, alpha, 0)
        elif f <= 52:
            # Moving down
            alpha = (f - 38) / 14.0
            current_y = int(round(E_y + 25 * alpha))
            roi = clean_arr[626:764, F_x:F_x+225].copy()
            y_offset = current_y - 626
            roi[y_offset:y_offset+113, :][mask_D] = green.astype(np.uint8)
            roi[y_offset:y_offset+113, :][mask_d_border] = [0, 0, 0]
            frame[626:764, F_x:F_x+225] = roi
        else:
            # Solid at F_y
            roi = clean_arr[626:764, F_x:F_x+225].copy()
            y_offset = F_y - 626
            roi[y_offset:y_offset+113, :][mask_D] = green.astype(np.uint8)
            roi[y_offset:y_offset+113, :][mask_d_border] = [0, 0, 0]
            frame[626:764, F_x:F_x+225] = roi

        frames.append(frame)

    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    solve()
