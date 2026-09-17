import cv2
import numpy as np
import imageio
import os

def create_video():
    first_frame = cv2.imread('/app/first_frame.png')
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB) # imageio expects RGB

    frames = []

    # Phase 1: Fade out
    # Target shape 4 is at x=355 to 460, y=450 to 575
    # Frames 0 to 22 (23 frames)
    for i in range(23):
        alpha = 1.0 - (i / 22.0)
        frame = first_frame.copy()
        work_area = frame[450:575, 355:460]
        
        # Blend with white
        white = np.full_like(work_area, 255)
        # We can't use cv2.addWeighted directly if alpha is not between 0 and 1, but it is.
        # Alternatively, direct numpy blending (ensure float math then cast to uint8)
        blended = (work_area.astype(np.float32) * alpha + white.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        
        frame[450:575, 355:460] = blended
        frames.append(frame)

    # Base image for sliding phase
    # Has shape 4 fully white (which is frame 22)
    base_img = frames[-1].copy()
    
    # Area that will be moved: shapes 5-9 (x=460 to 1024)
    # However, to easily handle the translation of a patch, we can take the whole right side 
    # from x=355 to 1024, set the shape 4 region to white, and warp it left.
    work_area_clean = base_img[450:575, 355:1024].copy()
    # Ensure region of shape 4 is perfectly white in the work area
    work_area_clean[:, :105] = 255

    # Phase 2: Slide left
    # Frames 23 to 44 (22 frames)
    for i in range(1, 23):
        t = i / 22.0
        s = t * 105.0
        
        # Translation matrix to move left by s
        M = np.float32([[1, 0, -s], [0, 1, 0]])
        
        # Apply affine transformation. borderValue is white to fill in the right side.
        shifted = cv2.warpAffine(work_area_clean, M, (work_area_clean.shape[1], work_area_clean.shape[0]), borderValue=(255,255,255))
        
        frame = base_img.copy()
        frame[450:575, 355:1024] = shifted
        frames.append(frame)

    os.makedirs('/app/output', exist_ok=True)
    
    # Save video
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == '__main__':
    create_video()
