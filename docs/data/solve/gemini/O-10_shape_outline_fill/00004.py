import cv2
import numpy as np
from PIL import Image, ImageDraw
import imageio

def make_video():
    orig_img = cv2.imread('/app/first_frame.png')
    if orig_img is None:
        raise ValueError("Could not read /app/first_frame.png")

    frames = []
    
    # Frame 0
    frames.append(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    
    # Pre-compute patch_solid
    patch_solid = Image.new('RGB', (230, 230), (255, 255, 255))
    draw = ImageDraw.Draw(patch_solid)
    draw.ellipse([34, 38, 194, 198], outline=(70, 153, 53), width=4)
    patch_solid_rgb = np.array(patch_solid)
    
    patch_orig_rgb = cv2.cvtColor(orig_img[650:880, 750:980], cv2.COLOR_BGR2RGB)
    orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    # Frames 1 to 10
    for t in range(1, 11):
        alpha = t / 10.0
        # crossfade
        patch_blend = cv2.addWeighted(patch_orig_rgb, 1 - alpha, patch_solid_rgb, alpha, 0)
        frame = orig_rgb.copy()
        frame[650:880, 750:980] = patch_blend
        frames.append(frame)
        
    # Frames 11 to 49
    for t in range(11, 50):
        t_dash = t - 10
        ratio = t_dash / 39.0
        dash_ratio = 1.0 - 0.25 * ratio
        
        patch_dashed = Image.new('RGB', (230, 230), (255, 255, 255))
        draw = ImageDraw.Draw(patch_dashed)
        gap_size = (1.0 - dash_ratio) * (360 / 64)
        for i in range(64):
            start_angle = i * (360 / 64) + gap_size / 2
            end_angle = (i + 1) * (360 / 64) - gap_size / 2
            draw.arc([34, 38, 194, 198], start=start_angle, end=end_angle, fill=(70, 153, 53), width=4)
            
        frame = orig_rgb.copy()
        frame[650:880, 750:980] = np.array(patch_dashed)
        frames.append(frame)
        
    # Frames 50 to 59
    for t in range(50, 60):
        frames.append(frames[-1].copy())
        
    # write video
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    for f in frames:
        writer.append_data(f)
    writer.close()
    
    print(f"Video saved with {len(frames)} frames.")

if __name__ == "__main__":
    make_video()
