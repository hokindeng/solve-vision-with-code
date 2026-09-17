import cv2
import numpy as np
import imageio
import os

def create_video():
    img_path = '/app/first_frame.png'
    first_frame = cv2.imread(img_path)
    
    if first_frame is None:
        raise ValueError("Could not read first_frame.png")

    final_frame = first_frame.copy()
    
    # We copy the 161x161 region from the first shape to the fifth shape.
    # The first shape is at X: 34 to 190. We take the 161x161 region around it: X: 32 to 193.
    # The fifth shape is at X: 832 to 993, Y: 432 to 593.
    source_region = first_frame[432:593, 32:193].copy()
    final_frame[432:593, 832:993] = source_region
    
    output_path = '/app/output/video.mp4'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fps = 16
    duration = 4.0
    num_frames = int(fps * duration)
    
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    for i in range(num_frames):
        t = i / (num_frames - 1)
        # Ease-in-out sine
        alpha = 0.5 - 0.5 * np.cos(np.pi * t)
        
        frame = first_frame.copy()
        
        blend_crop = cv2.addWeighted(
            first_frame[432:593, 832:993], 1 - alpha,
            final_frame[432:593, 832:993], alpha, 0
        )
        frame[432:593, 832:993] = blend_crop
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    create_video()
