import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise FileNotFoundError("Could not read /app/first_frame.png")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Find green components
    mask_green = cv2.inRange(img_rgb, np.array([0, 200, 0]), np.array([0, 200, 0]))
    num_labels_g, labels_g, stats_g, _ = cv2.connectedComponentsWithStats(mask_green, connectivity=8)

    # We want the green cone. We identified it as the downward pointing triangle,
    # which has the same area (~15051) as the orange cone.
    # The other green object is smaller (~9730) and points up.
    green_cone_label = -1
    for i in range(1, num_labels_g):
        if stats_g[i, cv2.CC_STAT_AREA] > 12000:
            green_cone_label = i
            break
    
    if green_cone_label == -1:
        # Fallback if sizes are different for some reason
        green_cone_label = 1

    # Find orange components
    mask_orange = cv2.inRange(img_rgb, np.array([255, 128, 0]), np.array([255, 128, 0]))
    num_labels_o, labels_o, stats_o, _ = cv2.connectedComponentsWithStats(mask_orange, connectivity=8)
    orange_cone_label = 1 # Assuming only 1 orange object

    mask_to_fade_green = (labels_g == green_cone_label)
    mask_to_fade_orange = (labels_o == orange_cone_label)

    os.makedirs('/app/output', exist_ok=True)
    out_path = '/app/output/video.mp4'

    n_frames = 96
    frames = []

    color_g = np.array([0, 200, 0], dtype=float)
    color_o = np.array([255, 128, 0], dtype=float)
    color_w = np.array([255, 255, 255], dtype=float)

    for i in range(n_frames):
        alpha = i / (n_frames - 1)
        
        frame = img_rgb.copy()
        
        cur_g = color_g * (1 - alpha) + color_w * alpha
        cur_o = color_o * (1 - alpha) + color_w * alpha
        
        frame[mask_to_fade_green] = np.round(cur_g).astype(np.uint8)
        frame[mask_to_fade_orange] = np.round(cur_o).astype(np.uint8)
        
        frames.append(frame)

    writer = imageio.get_writer(
        out_path, 
        fps=16, 
        codec='libx264', 
        pixelformat='yuv420p', 
        macro_block_size=None
    )
    for f in frames:
        writer.append_data(f)
    writer.close()
    
if __name__ == '__main__':
    solve()
