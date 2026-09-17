import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    bg_color = np.array([255, 255, 255])
    
    # The target symbol (red solid triangle) is at position 5 (rightmost).
    # X bounds for the 5th symbol are approximately 674 to 770.
    # We will use 670 to 1024 to be safe since it's the rightmost object and the rest is background.
    
    frames = []
    
    # 16 fps, about 45 frames (2.81 s)
    total_frames = 45
    fade_duration = 20
    
    for i in range(total_frames):
        frame = img.copy()
        
        # Phase 1: Fade out (frames 0 to fade_duration)
        if i <= fade_duration:
            alpha = i / float(fade_duration)
        else:
            alpha = 1.0
            
        region = frame[:, 670:]
        blended = region * (1 - alpha) + bg_color * alpha
        frame[:, 670:] = blended.astype(np.uint8)
        
        # Phase 2: Slide remaining objects
        # "then the remaining symbols slide leftward to close the gap"
        # Since the 5th symbol is the rightmost one, there are no symbols to its right to slide leftward.
        # Thus, no movement is applied to the remaining symbols.
        
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, macro_block_size=1, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    solve()
