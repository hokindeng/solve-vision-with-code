import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    
    # Background: copy image, erase domino 0 and 1
    bg = img.copy()
    for x in [139, 205]:
        bg[617:748, x:x+39] = [255, 255, 255]
        bg[746:751, x:x+39] = [85, 115, 139] # shadow line
    
    # Domino 0
    domino0 = img[617:748, 139:139+39].copy()
    # Domino 1
    domino1 = img[617:748, 205:205+39].copy()
    
    def draw_rotated_domino(frame, domino_img, pivot_x, pivot_y, angle_deg, offset_x, offset_y):
        h, w = domino_img.shape[:2]
        
        mask = np.zeros_like(domino_img[:,:,0])
        mask.fill(255)
        
        padded_domino = np.zeros((1024, 1024, 3), dtype=np.uint8)
        padded_mask = np.zeros((1024, 1024), dtype=np.uint8)
        
        padded_domino[offset_y:offset_y+h, offset_x:offset_x+w] = domino_img
        padded_mask[offset_y:offset_y+h, offset_x:offset_x+w] = mask
        
        M = cv2.getRotationMatrix2D((float(pivot_x), float(pivot_y)), -angle_deg, 1.0)
        
        rotated_domino = cv2.warpAffine(padded_domino, M, (1024, 1024), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        rotated_mask = cv2.warpAffine(padded_mask, M, (1024, 1024), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        idx = rotated_mask > 128
        frame[idx] = rotated_domino[idx]

    frames_count = 50
    start_f = 5
    end_f = 45
    
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    
    for f in range(frames_count):
        if f < start_f:
            t = 0.0
        elif f > end_f:
            t = 1.0
        else:
            t = (f - start_f) / (end_f - start_f)
        
        t_eased = t ** 1.5
        
        theta0 = 72.68 * t_eased
        if theta0 < 11.9:
            theta1 = 0.0
        else:
            t_rem = (theta0 - 11.9) / (72.68 - 11.9)
            theta1 = 90.0 * (t_rem ** 1.2)
        
        frame = bg.copy()
        
        # Draw domino 1 first (so it's behind domino 0 if they overlap)
        draw_rotated_domino(frame, domino1, pivot_x=244, pivot_y=748, angle_deg=theta1, offset_x=205, offset_y=617)
        # Draw domino 0
        draw_rotated_domino(frame, domino0, pivot_x=178, pivot_y=748, angle_deg=theta0, offset_x=139, offset_y=617)
        
        # imageio expects RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    solve()
