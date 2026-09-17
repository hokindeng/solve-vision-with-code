import cv2
import numpy as np
import imageio
import os

def solve():
    os.makedirs('/app/output', exist_ok=True)
    
    img_orig = cv2.imread('/app/first_frame.png')
    D_crop = img_orig[665:700, 110:251].copy()

    E_center_x, E_center_y = 518.5, 682.5
    F_center_x, F_center_y = 854.5, 682.5

    start_color = np.array([153, 89, 76], dtype=float)
    end_color = np.array([30, 153, 153], dtype=float)

    frames = []
    num_frames = 60
    
    for f in range(num_frames):
        if f == 0:
            frames.append(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
            continue
            
        img = img_orig.copy()
        
        if f < 30:
            # Phase 1: 1 to 29
            t = f / 29.0
            cur_color = start_color + (end_color - start_color) * t
            
            # Erase E question mark
            img[600:750, 400:650] = 255
            
            # Draw E
            E_img = D_crop.copy()
            mask = np.all(E_img == [153, 89, 76], axis=-1)
            E_img[mask] = cur_color.astype(np.uint8)
            
            h, w = E_img.shape[:2]
            x = int(round(E_center_x - w / 2))
            y = int(round(E_center_y - h / 2))
            img[y:y+h, x:x+w] = E_img
            
        else:
            # Phase 2: 30 to 59
            img[600:750, 400:650] = 255
            img[600:750, 720:980] = 255
            
            # Draw E final
            E_img = D_crop.copy()
            mask = np.all(E_img == [153, 89, 76], axis=-1)
            E_img[mask] = end_color.astype(np.uint8)
            h, w = E_img.shape[:2]
            x = int(round(E_center_x - w / 2))
            y = int(round(E_center_y - h / 2))
            img[y:y+h, x:x+w] = E_img
            
            # Draw F
            t = (f - 30) / 29.0
            scale = 1.0 - 0.5 * t
            
            if scale == 1.0:
                F_img = E_img.copy()
            else:
                F_img = cv2.resize(E_img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            
            h, w = F_img.shape[:2]
            x = int(round(F_center_x - w / 2))
            y = int(round(F_center_y - h / 2))
            img[y:y+h, x:x+w] = F_img
            
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    solve()
