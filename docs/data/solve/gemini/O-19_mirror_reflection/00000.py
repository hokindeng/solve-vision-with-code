import cv2
import numpy as np
import imageio
import os

def draw_arrowhead(img, tip_x, tip_y, tail_x, tail_y, color, thickness=3, length=31, angle=30):
    rev_angle = np.arctan2(tail_y - tip_y, tail_x - tip_x)
    a1 = rev_angle + np.radians(angle)
    a2 = rev_angle - np.radians(angle)
    
    p1 = (int(tip_x + length * np.cos(a1)), int(tip_y + length * np.sin(a1)))
    p2 = (int(tip_x + length * np.cos(a2)), int(tip_y + length * np.sin(a2)))
    
    cv2.line(img, (int(tip_x), int(tip_y)), p1, color, thickness, cv2.LINE_AA)
    cv2.line(img, (int(tip_x), int(tip_y)), p2, color, thickness, cv2.LINE_AA)

def main():
    img = cv2.imread('/app/first_frame.png')
    
    # Prepare base image by removing the angle annotation on the right
    base = img.copy()
    for y in range(530, 595):
        for x in range(596, 720):
            c = base[y, x].astype(int)
            # Keep normal line (around 150, 150, 150)
            if abs(c[0] - 150) < 30 and abs(c[1] - 150) < 30 and abs(c[2] - 150) < 30 and abs(c[0]-c[1])<10:
                continue
            # Keep blue ray and its anti-aliasing (higher blue than red/green)
            if c[0] > c[1] + 20 and c[0] > c[2] + 20:
                continue
            base[y, x] = [255, 255, 255]
            
    # Remove the black arc segment crossing the blue ray and right of it
    for y in range(550, 560):
        for x in range(590, 598):
            c = base[y, x].astype(int)
            if c[0] < 50 and c[1] < 50 and c[2] < 50:
                base[y, x] = [255, 255, 255]
                
    # Restore the blue ray where the arc crossed it
    for y in [554, 555]:
        base[y, 592:595] = [255, 0, 0]
        # blend edges slightly to match anti-aliasing
        base[y, 591] = [255, 128, 128]
        base[y, 595] = [255, 128, 128]

    frames = []
    
    # Frame 0: exactly first_frame.png
    frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Ray parameters
    start_x, start_y = 596.0, 595.0
    # dy = -595 to reach y=0
    # dx = 57 / 545 * 595 = 62.22
    end_x, end_y = start_x + 62.22, 0.0
    
    alpha_mult = 0.71
    
    for i in range(1, 35):
        progress = i / 34.0
        cur_x = start_x + (end_x - start_x) * progress
        cur_y = start_y + (end_y - start_y) * progress
        
        blank = np.zeros((1024, 1024, 4), dtype=np.uint8)
        cv2.line(blank, (int(round(start_x)), int(round(start_y))), (int(round(cur_x)), int(round(cur_y))), (255, 0, 0, 255), 3, cv2.LINE_AA)
        
        ray_len = np.hypot(cur_x - start_x, cur_y - start_y)
        arrow_len = min(31, ray_len * 0.8)
        
        if arrow_len > 0:
            draw_arrowhead(blank, cur_x, cur_y, start_x, start_y, (255, 0, 0, 255), 3, arrow_len, 30)
            
        alpha = (blank[:, :, 3] / 255.0) * alpha_mult
        
        frame = base.copy().astype(float)
        for c_idx in range(3):
            frame[:, :, c_idx] = frame[:, :, c_idx] * (1 - alpha) + blank[:, :, c_idx] * alpha
        
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, macro_block_size=None, pixelformat='yuv420p', codec='libx264')

if __name__ == '__main__':
    main()
