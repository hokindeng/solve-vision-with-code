import cv2
import numpy as np
import imageio
import os

def main():
    img_path = '/app/first_frame.png'
    out_path = '/app/output/video.mp4'
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Load the first frame
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read {img_path}")
        
    # Extract unique colors and find specific elements by their known pixel counts
    # from the first frame.
    unique_colors, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
    
    bg_bgr = None
    min_bgr = None
    hour_bgr = None
    center_bgr = None
    
    for c, count in zip(unique_colors, counts):
        if count == 1385:
            min_bgr = c
        elif count == 1553:
            hour_bgr = c
        elif count == 349:
            center_bgr = c
        elif count > 1000000:
            bg_bgr = c

    # Create masks for the movable parts
    min_mask = np.all(img == min_bgr, axis=-1).astype(np.uint8) * 255
    hour_mask = np.all(img == hour_bgr, axis=-1).astype(np.uint8) * 255
    center_mask = np.all(img == center_bgr, axis=-1).astype(np.uint8) * 255

    # Create a clean background by erasing the hands and center dot
    clean_bg = img.copy()
    clean_bg[min_mask > 0] = bg_bgr
    clean_bg[hour_mask > 0] = bg_bgr
    clean_bg[center_mask > 0] = bg_bgr

    # Set up video writer
    writer = imageio.get_writer(
        out_path, 
        fps=16, 
        codec='libx264', 
        pixelformat='yuv420p', 
        macro_block_size=None
    )

    num_frames = 120
    center_pt = (512, 512)

    for i in range(num_frames):
        # Time fraction from 0.0 to 1.0
        t = i / (num_frames - 1)
        
        # 15 hours total time elapsed
        # Minute hand: 15 full rotations (1 rotation = 360 degrees)
        angle_minute_deg = t * 15 * 360
        # Hour hand: 15 hours (1 hour = 30 degrees)
        angle_hour_deg = t * 15 * 30

        # Rotate clockwise (negative angle in OpenCV means clockwise when y points down)
        M_min = cv2.getRotationMatrix2D(center_pt, -angle_minute_deg, 1.0)
        M_hour = cv2.getRotationMatrix2D(center_pt, -angle_hour_deg, 1.0)

        rot_min_mask = cv2.warpAffine(min_mask, M_min, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)
        rot_hour_mask = cv2.warpAffine(hour_mask, M_hour, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)

        # Draw the frame
        frame = clean_bg.copy()
        
        # Draw hour hand first, then minute hand on top
        frame[rot_hour_mask > 0] = hour_bgr
        frame[rot_min_mask > 0] = min_bgr
        
        # Draw the center dot on top of both hands
        frame[center_mask > 0] = center_bgr

        # Convert to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)

    writer.close()
    print(f"Video saved to {out_path}")

if __name__ == "__main__":
    main()
