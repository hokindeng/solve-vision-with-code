import cv2
import numpy as np
import imageio
import math
import os

def ease_in_out(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)

def main():
    img = cv2.imread('/app/first_frame.png')
    
    # 1. Extract ball
    ball_mask = np.logical_or(np.all(img == [100, 25, 175], axis=-1), np.all(img == [180, 105, 255], axis=-1)).astype(np.uint8) * 255
    x_ball, y_ball, w_ball, h_ball = cv2.boundingRect(ball_mask)
    ball_patch = img[y_ball:y_ball+h_ball, x_ball:x_ball+w_ball].copy()
    ball_patch_mask = ball_mask[y_ball:y_ball+h_ball, x_ball:x_ball+w_ball] == 255
    
    # 2. Reconstruct background
    bg = img.copy()
    bg[ball_mask == 255] = [255, 255, 255]
    
    p3_x, p3_y = 423, 563
    p3_patch = img[p3_y:p3_y+20, p3_x:p3_x+20].copy()
    p3_mask = np.any(p3_patch != [255, 255, 255], axis=-1)
    
    platforms = [
        (371, 604), (389, 590), (406, 576)
    ]
    
    # Restore p0, p1, p2 only inside the original ball mask
    for px, py in platforms:
        for r in range(20):
            for c in range(20):
                if p3_mask[r, c]:
                    # Check if the pixel was covered by the ball originally
                    # If it was covered by the ball, we replace it with the platform pixel.
                    # Since we want to handle the case where the ball moves and reveals it,
                    # we only replace pixels that were originally covered by the ball.
                    if 0 <= py+r < 1024 and 0 <= px+c < 1024:
                        if ball_mask[py+r, px+c] == 255:
                            bg[py+r, px+c] = p3_patch[r, c]
                            
    # Ball start and end coordinates (top-left)
    start_x = 340.0
    start_y = 574.0
    end_x = 601.0
    end_y = 371.0
    
    # Generate frames
    frames = []
    num_frames = 64
    for i in range(num_frames):
        t = i / (num_frames - 1)
        eased_t = ease_in_out(t)
        
        nx = int(round(start_x + eased_t * (end_x - start_x)))
        ny = int(round(start_y + eased_t * (end_y - start_y)))
        
        frame = bg.copy()
        
        for r in range(h_ball):
            for c in range(w_ball):
                if ball_patch_mask[r, c]:
                    if 0 <= ny+r < 1024 and 0 <= nx+c < 1024:
                        frame[ny+r, nx+c] = ball_patch[r, c]
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    main()
