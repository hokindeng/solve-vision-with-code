import cv2
import numpy as np
import imageio

def main():
    # Read the first frame
    img = cv2.imread('/app/first_frame.png')
    final_img = img.copy()

    # The grid is 3x3 on a 1024x1024 image
    # We copy the top-right cell (cell 0,2) content to the bottom-right cell (cell 2,2)
    # to complete the pattern. The inner region is used to avoid touching the grid lines.
    y_start_src = 10
    y_end_src = 330
    x_start = 682 + 10
    x_end = 1024 - 10

    y_start_dst = 682 + 10
    y_end_dst = 682 + 330

    final_img[y_start_dst:y_end_dst, x_start:x_end] = img[y_start_src:y_end_src, x_start:x_end]

    num_frames = 35
    fps = 16

    # Create video writer
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, codec='libx264', macro_block_size=None, pixelformat='yuv420p')

    # Animate the transition (alpha blend)
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        
        current_frame = img.copy()
        
        # Blend only the modified region
        current_region = cv2.addWeighted(
            final_img[y_start_dst:y_end_dst, x_start:x_end], alpha,
            img[y_start_dst:y_end_dst, x_start:x_end], 1 - alpha,
            0
        )
        current_frame[y_start_dst:y_end_dst, x_start:x_end] = current_region
        
        # Convert BGR to RGB for imageio
        current_frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        writer.append_data(current_frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    main()
