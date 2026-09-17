import cv2
import numpy as np
import imageio

def make_video():
    first_frame = cv2.imread('/app/first_frame.png')
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    # Extract red plus patch
    plus_patch = first_frame[602:602+161, 115:115+161].copy()
    # In RGB, red is [255, 0, 0]
    mask_red = (plus_patch == [255, 0, 0]).all(axis=-1)

    color_red = np.array([255, 0, 0], dtype=np.float32)
    # The green minus is at 402, 321
    # Let's dynamically get the green color from the green minus just to be perfectly accurate
    green_patch = first_frame[321:321+41, 402:402+161]
    # Find the most common non-background color in green minus
    unique_colors, counts = np.unique(green_patch.reshape(-1, 3), axis=0, return_counts=True)
    green_color_rgb = None
    max_count = 0
    for c, count in zip(unique_colors, counts):
        if not np.array_equal(c, [255, 255, 255]) and not np.array_equal(c, [0, 0, 0]):
            if count > max_count:
                max_count = count
                green_color_rgb = c
    
    color_green = green_color_rgb.astype(np.float32)

    patch_green = plus_patch.copy()
    patch_green[mask_red] = color_green

    frames = []
    # Frame 0
    frames.append(first_frame.copy())

    # Phase 1: 1 to 29
    for i in range(1, 30):
        t = (i - 1) / 28.0
        curr_color = color_red * (1 - t) + color_green * t
        curr_patch = plus_patch.copy()
        curr_patch[mask_red] = curr_color.astype(np.uint8)
        
        frame = first_frame.copy()
        frame[660:690, 469:496] = 255 # erase E's ?
        frame[602:602+161, 402:402+161] = curr_patch
        frames.append(frame)

    # Phase 2: 30 to 59
    for i in range(30, 60):
        t = (i - 30) / 29.0
        # move down by 100 pixels
        offset = int(100 * t)
        
        frame = first_frame.copy()
        frame[660:690, 469:496] = 255 # erase E's ?
        frame[660:690, 756:783] = 255 # erase F's ?
        
        frame[602:602+161, 402:402+161] = patch_green
        frame[602+offset:602+offset+161, 689:689+161] = patch_green
        frames.append(frame)

    import os
    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, macro_block_size=None, codec='libx264', format='FFMPEG', pixelformat='yuv420p')

if __name__ == '__main__':
    make_video()
