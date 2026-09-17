import cv2
import numpy as np
import imageio

def ease_in_out(t):
    # Smooth step
    return t * t * (3 - 2 * t)

def main():
    # Load first frame
    first_frame = cv2.imread('/app/first_frame.png')
    
    # Tile positions
    tiles_info = [
        (276, 276), # T0 (y, x)
        (276, 526), # T1
        (526, 276), # T2
        (526, 526)  # T3
    ]
    
    # Extract inner squares
    inners = []
    for (y, x) in tiles_info:
        inners.append(first_frame[y+4:y+219, x+4:x+219].copy())
        
    target_angles = [0, 0, 0, -90]
    
    # Video writer
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', macro_block_size=None, quality=8, pixelformat='yuv420p')
    
    num_frames = 96
    
    for f in range(num_frames):
        # Calculate progress
        if num_frames > 1:
            t = f / (num_frames - 1)
        else:
            t = 1.0
        
        progress = ease_in_out(t)
        
        frame = first_frame.copy()
        
        for i in range(4):
            angle = target_angles[i] * progress
            if angle != 0:
                # Rotate the inner square
                # Center is 107.0 for a 215x215 square
                M = cv2.getRotationMatrix2D((107.0, 107.0), angle, 1.0)
                rotated = cv2.warpAffine(inners[i], M, (215, 215), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
                
                # Paste it back
                y, x = tiles_info[i]
                frame[y+4:y+219, x+4:x+219] = rotated
            else:
                pass # Already there
                
        # Write frame (convert BGR to RGB)
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    writer.close()

if __name__ == '__main__':
    main()
