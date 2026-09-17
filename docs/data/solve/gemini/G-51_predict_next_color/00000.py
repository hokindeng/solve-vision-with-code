import numpy as np
from PIL import Image
import imageio
import os

def main():
    # Load the first frame
    img_path = '/app/first_frame.png'
    img = np.array(Image.open(img_path))
    
    # The sequence of colors is Green, Blue, Purple, Green.
    # The next color should be Blue.
    # We will copy the Blue circle patch and blend it over the Gray placeholder.
    
    # Blue patch bounding box (including enough background to cover the gray border):
    # Blue shape is centered at x=311.65. Gray placeholder is centered at x=911.65.
    # Delta x is exactly 600.
    # Gray box bounds: y: 430 to 594, x: 840 to 983
    # Corresponding Blue box bounds: y: 430 to 594, x: 240 to 384
    
    start_y, end_y = 430, 595
    start_x_blue, end_x_blue = 240, 384
    start_x_gray, end_x_gray = 840, 984
    
    blue_patch = img[start_y:end_y, start_x_blue:end_x_blue].copy()
    gray_patch = img[start_y:end_y, start_x_gray:end_x_gray].copy()
    
    frames = []
    num_frames = 64
    
    for t in range(num_frames):
        alpha = t / (num_frames - 1)
        
        # Blend patches
        blended = (blue_patch.astype(float) * alpha + gray_patch.astype(float) * (1 - alpha)).astype(np.uint8)
        
        frame = img.copy()
        frame[start_y:end_y, start_x_gray:end_x_gray] = blended
        frames.append(frame)
        
    # Save video
    os.makedirs('/app/output', exist_ok=True)
    out_path = '/app/output/video.mp4'
    imageio.mimwrite(
        out_path, 
        frames, 
        fps=16, 
        codec='libx264', 
        pixelformat='yuv420p'
    )

if __name__ == '__main__':
    main()
