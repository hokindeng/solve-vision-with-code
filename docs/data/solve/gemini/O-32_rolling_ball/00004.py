import cv2
import numpy as np
from PIL import Image
import os
import imageio

def solve():
    img = cv2.imread('/app/first_frame.png')
    
    # 1. Create clean background
    # Ball colors are [0, 60, 175] and [0, 140, 255] in BGR
    mask1 = cv2.inRange(img, np.array([0, 60, 175]), np.array([0, 60, 175]))
    mask2 = cv2.inRange(img, np.array([0, 140, 255]), np.array([0, 140, 255]))
    m = cv2.bitwise_or(mask1, mask2)
    
    bg_cv = img.copy()
    bg_cv[m > 0] = [255, 255, 255]
    bg_img = Image.fromarray(cv2.cvtColor(bg_cv, cv2.COLOR_BGR2RGB))
    
    # 2. Extract ball and pad it
    ball_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    ball_rgba[:, :, 3] = m
    ball_crop = ball_rgba[510:591, 310:391]
    
    pad = 15
    padded_ball = np.zeros((81 + 2 * pad, 81 + 2 * pad, 4), dtype=np.uint8)
    padded_ball[pad:pad+81, pad:pad+81] = ball_crop
    
    ball_img = Image.fromarray(cv2.cvtColor(padded_ball, cv2.COLOR_BGRA2RGBA))
    
    # 3. Path calculation
    # Platforms form a perfect parabola
    pts = np.array([
        [391.2, 569.4], [411.0, 572.0], [432.5, 571.5], [454.2, 566.9], 
        [474.5, 560.5], [495.0, 552.0], [514.7, 542.4], [534.1, 532.9], 
        [553.5, 522.9], [573.0, 513.0], [593.0, 504.0], [612.5, 495.0], 
        [632.8, 487.6], [653.5, 480.5], [674.0, 473.5]
    ])
    p = np.polyfit(pts[:,0], pts[:,1], 2)
    
    # Offset of the ball center from the parabola
    # Initial ball center is at (350, 550)
    start_x = 350.0
    start_y = 550.0
    end_x = 674.0
    offset = np.polyval(p, start_x) - start_y
    
    # 4. Generate frames
    os.makedirs('/app/output', exist_ok=True)
    frames = []
    
    num_frames = 64
    for i in range(num_frames):
        t = i / float(num_frames - 1)
        
        # Ease out cubic interpolation
        ease_t = 1 - (1 - t)**3
        
        x = start_x + (end_x - start_x) * ease_t
        y = np.polyval(p, x) - offset
        
        frame = bg_img.copy()
        
        # Distance traveled for rotation
        dist = x - start_x
        # Radius is 40.0
        angle_rad = dist / 40.0
        angle_deg = np.degrees(angle_rad)
        
        rotated_ball = ball_img.rotate(-angle_deg, resample=Image.BICUBIC)
        
        # Paste ball
        paste_x = int(round(x - 40.0 - pad))
        paste_y = int(round(y - 40.0 - pad))
        
        frame.paste(rotated_ball, (paste_x, paste_y), rotated_ball)
        frames.append(np.array(frame))
        
    # 5. Write video
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, quality=8, macro_block_size=None)

if __name__ == '__main__':
    solve()
