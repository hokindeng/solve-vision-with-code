import numpy as np
from PIL import Image
import imageio
import os

def generate_video():
    os.makedirs('/app/output', exist_ok=True)

    first_frame = Image.open('/app/first_frame.png').convert('RGB')
    base_arr = np.array(first_frame)

    # The green minus is D (x=115..276, y=662..703)
    green_minus = base_arr[662:703, 115:276].copy()
    mask = (green_minus == [70, 153, 53]).all(axis=-1)

    frames = []
    # Total frames = 60
    for i in range(60):
        if i == 0:
            frames.append(base_arr.copy())
            continue

        frame_arr = base_arr.copy()
        
        if i < 30:
            # Phase 1: Recolor E, keep ? at F
            # i goes from 1 to 29
            t = (i - 1) / 28.0 if i > 1 else 0.0
            
            # Erase first ?
            frame_arr[650:710, 460:500] = [255, 255, 255]
            
            # Interpolate color from green to purple
            current_color = [
                int(70 + t * (91 - 70)),
                int(153 + t * (30 - 153)),
                int(53 + t * (153 - 53))
            ]
            
            E = green_minus.copy()
            E[mask] = current_color
            
            # Draw E
            frame_arr[662:703, 402:563] = E
            
        else:
            # Phase 2: E is fully purple, F moves down
            # i goes from 30 to 59
            
            # Erase first ? and draw E as fully purple
            frame_arr[650:710, 460:500] = [255, 255, 255]
            E = green_minus.copy()
            E[mask] = [91, 30, 153]
            frame_arr[662:703, 402:563] = E
            
            # Erase second ? and clear path for F
            frame_arr[650:810, 689:850] = [255, 255, 255]
            
            # Interpolate position from y=662 to y=762
            t = (i - 30) / 29.0
            y_pos = int(662 + t * 100)
            
            # Draw F
            frame_arr[y_pos:y_pos+41, 689:850] = E

        frames.append(frame_arr)

    # Save to mp4
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', pixelformat='yuv420p')
    print("Video generated successfully.")

if __name__ == '__main__':
    generate_video()
