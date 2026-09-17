import cv2
import numpy as np
import os
import subprocess
import tempfile

def main():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise FileNotFoundError("Could not read /app/first_frame.png")

    # Find the two circles using their known colors in BGR
    # Color 1: Blue=48, Green=130, Red=245
    mask1 = np.all(img == (48, 130, 245), axis=-1)
    # Color 2: Blue=180, Green=105, Red=255
    mask2 = np.all(img == (180, 105, 255), axis=-1)

    # Reconstruct Circle 1 full shape (it was cut off at the top border)
    # The original center of Circle 1 is exactly at (350, 307)
    y1, x1 = np.where(mask1)
    dy1 = y1 - 307
    dx1 = x1 - 350
    # Mirror across Y-axis relative to center to recover the cut-off top half
    full_dy1 = np.concatenate([dy1, -dy1])
    full_dx1 = np.concatenate([dx1, dx1])
    pts1 = list(set(zip(full_dy1, full_dx1)))

    # Extract Circle 2 shape (fully visible)
    # The original center of Circle 2 is exactly at (834, 716)
    y2, x2 = np.where(mask2)
    dy2 = y2 - 716
    dx2 = x2 - 834
    pts2 = list(set(zip(dy2, dx2)))

    # Convert to fast arrays
    dy1_arr = np.array([p[0] for p in pts1])
    dx1_arr = np.array([p[1] for p in pts1])
    dy2_arr = np.array([p[0] for p in pts2])
    dx2_arr = np.array([p[1] for p in pts2])

    num_frames = 40
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for t in range(num_frames):
            # Interpolate centers
            # Start: C1(350, 307), C2(834, 716)
            # End: (512, 512) for both
            cx1 = int(round(350 + (512 - 350) * t / (num_frames - 1)))
            cy1 = int(round(307 + (512 - 307) * t / (num_frames - 1)))
            
            cx2 = int(round(834 + (512 - 834) * t / (num_frames - 1)))
            cy2 = int(round(716 + (512 - 716) * t / (num_frames - 1)))

            # Pure white background
            frame = np.full((1024, 1024, 3), 255, dtype=np.uint8)

            # Draw circle 1
            y_draw1 = cy1 + dy1_arr
            x_draw1 = cx1 + dx1_arr
            valid1 = (y_draw1 >= 0) & (y_draw1 < 1024) & (x_draw1 >= 0) & (x_draw1 < 1024)
            frame[y_draw1[valid1], x_draw1[valid1]] = (48, 130, 245)

            # Draw circle 2
            y_draw2 = cy2 + dy2_arr
            x_draw2 = cx2 + dx2_arr
            valid2 = (y_draw2 >= 0) & (y_draw2 < 1024) & (x_draw2 >= 0) & (x_draw2 < 1024)
            frame[y_draw2[valid2], x_draw2[valid2]] = (180, 105, 255)

            cv2.imwrite(os.path.join(temp_dir, f'frame_{t:04d}.png'), frame)

        # Encode to mp4
        os.makedirs('/app/output', exist_ok=True)
        out_path = '/app/output/video.mp4'
        cmd = [
            'ffmpeg', '-y', 
            '-framerate', '16', 
            '-i', os.path.join(temp_dir, 'frame_%04d.png'),
            '-c:v', 'libx264', 
            '-pix_fmt', 'yuv420p', 
            out_path
        ]
        # Run ffmpeg, suppressing output unless there's an error
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == '__main__':
    main()
