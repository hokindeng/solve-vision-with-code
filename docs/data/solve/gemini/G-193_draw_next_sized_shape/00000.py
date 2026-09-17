import cv2
import numpy as np
import imageio
import os

def solve():
    # Read the first frame
    first_frame = cv2.imread('/app/first_frame.png')
    if first_frame is None:
        raise ValueError("Could not read /app/first_frame.png")

    # The small shape (Component 15) is at bounding box x=419, y=493, w=39, h=39
    orig_patch = first_frame[493:532, 419:458]
    shape_mask = np.all(orig_patch == [212, 182, 6], axis=-1)

    # Target location for the new shape
    # Center of dashed box is at x=886, y=512
    # So top-left of 39x39 patch is x=867, y=493
    target_x = 867
    target_y = 493

    # We will use an expanding circular mask from the center of the patch
    Y, X = np.ogrid[0:39, 0:39]
    center_x, center_y = 19.0, 19.0
    dist = np.hypot(X - center_x, Y - center_y)

    num_frames = 60
    fps = 16
    max_radius = 19.5 # slightly larger than max distance in the 39x39 patch to fully reveal

    os.makedirs('/app/output', exist_ok=True)
    video_path = '/app/output/video.mp4'
    writer = imageio.get_writer(video_path, fps=fps, codec='libx264', macro_block_size=None, pixelformat='yuv420p')

    for i in range(num_frames):
        frame = first_frame.copy()
        
        if i > 0:
            current_r = max_radius * (i / (num_frames - 1))
            # Mask of pixels to reveal in this frame
            reveal_mask = shape_mask & (dist <= current_r)
            
            # Apply to target region
            target_patch = frame[target_y:target_y+39, target_x:target_x+39]
            target_patch[reveal_mask] = [212, 182, 6]
            
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)

    writer.close()
    print("Video generated at:", video_path)

if __name__ == '__main__':
    solve()
