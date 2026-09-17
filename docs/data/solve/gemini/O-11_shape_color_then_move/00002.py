import cv2
import numpy as np
import imageio
import os

def solve():
    first_frame = cv2.imread('/app/first_frame.png')
    
    # Colors in BGR format
    C1 = np.array([114, 229, 210], dtype=float)
    C2 = np.array([45, 168, 229], dtype=float)

    # Extract minus template from D
    # D is at x=115, y=662. size is 161x41
    minus_template = first_frame[662:703, 115:276].copy()
    minus_mask = np.all(minus_template == [114, 229, 210], axis=-1)

    frames = []
    
    # Generate exactly 60 frames
    for i in range(60):
        if i == 0:
            frame = first_frame.copy()
        else:
            frame = first_frame.copy()
            # Erase Q1 (at position E) by filling with white background
            frame[650:715, 450:510] = 255
            
            if i < 30:
                # Phase 1: Color change of E (Frames 1 to 29)
                t = (i - 1) / 28.0
                curr_color = C1 * (1 - t) + C2 * t
                curr_color = np.round(curr_color).astype(np.uint8)
                
                # Draw minus at E (x=402, y=662)
                minus_E = minus_template.copy()
                minus_E[minus_mask] = curr_color
                frame[662:703, 402:563] = minus_E
                
            else:
                # Phase 2: Move up of F (Frames 30 to 59)
                # E has reached final color and is static
                minus_E = minus_template.copy()
                minus_E[minus_mask] = C2.astype(np.uint8)
                frame[662:703, 402:563] = minus_E
                
                # Erase Q2 (at position F) by filling with white background
                frame[650:715, 740:800] = 255
                
                t = (i - 30) / 29.0
                curr_y = int(round(662 * (1 - t) + 602 * t))
                
                # Draw minus at F (x=689, varying y)
                minus_F = minus_template.copy()
                minus_F[minus_mask] = C2.astype(np.uint8)
                frame[curr_y:curr_y+41, 689:850] = minus_F
                
        # Convert BGR to RGB for imageio video writer
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    # Save output video
    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, macro_block_size=None, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    solve()
