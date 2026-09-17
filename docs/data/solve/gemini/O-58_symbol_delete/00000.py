import cv2
import numpy as np
import imageio
import os

def ease_in_out(t):
    return 3 * t**2 - 2 * t**3

def solve():
    img = cv2.imread('/app/first_frame.png')
    
    frames = []
    
    sym6_patch = img[464:561, 674:771].astype(np.float32)
    sym7_patch = img[464:561, 779:876].copy()
    
    base_frame = img.copy()
    base_frame[464:561, 674:771] = 255
    base_frame[464:561, 779:876] = 255
    
    # Frames 0 to 15 (16 frames) - Fade out sym 6
    for f in range(16):
        frame = base_frame.copy()
        
        # Sym 7 stays in place
        frame[464:561, 779:876] = sym7_patch
        
        # Sym 6 fades
        alpha = 1.0 - (f / 15.0)
        blended = sym6_patch * alpha + 255.0 * (1 - alpha)
        frame[464:561, 674:771] = blended.astype(np.uint8)
        
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    # Frames 16 to 35 (20 frames) - Slide sym 7 to sym 6 position
    for f in range(16, 36):
        frame = base_frame.copy()
        
        progress = (f - 15) / 20.0
        progress_eased = ease_in_out(progress)
        
        current_x = int(round(779 - progress_eased * (779 - 674)))
        
        frame[464:561, current_x:current_x+97] = sym7_patch
        
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    # Frames 36 to 44 (9 frames) - Hold
    for f in range(36, 45):
        frame = base_frame.copy()
        frame[464:561, 674:771] = sym7_patch
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    os.makedirs('/app/output', exist_ok=True)
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == '__main__':
    solve()
