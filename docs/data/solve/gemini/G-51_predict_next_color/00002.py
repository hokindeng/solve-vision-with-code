import cv2
import numpy as np
import imageio
import os

def main():
    first_frame = cv2.imread('/app/first_frame.png')
    if first_frame is None:
        raise ValueError("Could not read /app/first_frame.png")
        
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    crop1 = first_frame_rgb[434:591, 34:191].copy()
    crop5 = first_frame_rgb[434:591, 834:991].copy()

    frames = []
    num_frames = 64
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        blended_crop = crop5.astype(float) * (1 - alpha) + crop1.astype(float) * alpha
        
        frame = first_frame_rgb.copy()
        frame[434:591, 834:991] = blended_crop.astype(np.uint8)
        
        frames.append(frame)

    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == '__main__':
    main()
