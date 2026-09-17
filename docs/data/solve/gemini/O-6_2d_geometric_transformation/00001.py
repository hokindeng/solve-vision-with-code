import cv2
import numpy as np
import imageio
from scipy.ndimage import distance_transform_edt

def find_rotation_center(img_rgb):
    c0_mask = (img_rgb == [0, 0, 0]).all(axis=2).astype(np.uint8)
    coords = np.argwhere(c0_mask)
    if len(coords) == 0:
        return 572, 589 # fallback
    cy = int(np.round(coords[:, 0].mean()))
    cx = int(np.round(coords[:, 1].mean()))
    return cx, cy

def find_target_angle(img_rgb, Cx, Cy):
    outline_mask = (img_rgb == [50, 50, 50]).all(axis=2).astype(np.uint8)
    dashed_mask = (img_rgb == [100, 100, 100]).all(axis=2).astype(np.uint8)
    dashed_pixels = dashed_mask > 0

    best_angle = 0
    min_dist = float('inf')

    # coarse search
    for angle in np.arange(0, 360, 5.0):
        M = cv2.getRotationMatrix2D((Cx, Cy), angle, 1.0)
        rotated_outline = cv2.warpAffine(outline_mask, M, (img_rgb.shape[1], img_rgb.shape[0]), flags=cv2.INTER_NEAREST)
        if rotated_outline.sum() == 0:
            continue
        dist_map = distance_transform_edt(1 - rotated_outline)
        dist = dist_map[dashed_pixels].mean()
        
        if dist < min_dist:
            min_dist = dist
            best_angle = angle

    # medium search
    coarse_best = best_angle
    min_dist = float('inf')
    for angle in np.arange(coarse_best - 5.0, coarse_best + 5.0, 1.0):
        M = cv2.getRotationMatrix2D((Cx, Cy), angle, 1.0)
        rotated_outline = cv2.warpAffine(outline_mask, M, (img_rgb.shape[1], img_rgb.shape[0]), flags=cv2.INTER_NEAREST)
        if rotated_outline.sum() == 0:
            continue
        dist_map = distance_transform_edt(1 - rotated_outline)
        dist = dist_map[dashed_pixels].mean()
        
        if dist < min_dist:
            min_dist = dist
            best_angle = angle
            
    # fine search
    medium_best = best_angle
    min_dist = float('inf')
    for angle in np.arange(medium_best - 1.0, medium_best + 1.0, 0.1):
        M = cv2.getRotationMatrix2D((Cx, Cy), angle, 1.0)
        rotated_outline = cv2.warpAffine(outline_mask, M, (img_rgb.shape[1], img_rgb.shape[0]), flags=cv2.INTER_NEAREST)
        if rotated_outline.sum() == 0:
            continue
        dist_map = distance_transform_edt(1 - rotated_outline)
        dist = dist_map[dashed_pixels].mean()
        
        if dist < min_dist:
            min_dist = dist
            best_angle = angle

    return best_angle

def generate_video():
    img = cv2.imread('/app/first_frame.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    Cx, Cy = find_rotation_center(img_rgb)
    print(f"Rotation center: ({Cx}, {Cy})")
    
    target_angle = find_target_angle(img_rgb, Cx, Cy)
    print(f"Calculated target angle: {target_angle}")
    
    moving_mask = ((img_rgb == [77, 147, 189]).all(axis=2) | (img_rgb == [50, 50, 50]).all(axis=2))
    clean_bg = img_rgb.copy()
    clean_bg[moving_mask] = [240, 240, 240]

    # Pre-multiplied alpha for flawless interpolation
    moving_rgba = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.float32)
    moving_rgba[..., :3] = img_rgb * moving_mask[:, :, None]
    moving_rgba[..., 3] = moving_mask.astype(np.float32) * 255.0

    num_frames = 70
    fps = 16

    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=None)

    for i in range(num_frames):
        angle = target_angle * (i / (num_frames - 1))
        
        M = cv2.getRotationMatrix2D((Cx, Cy), angle, 1.0)
        rot_rgba = cv2.warpAffine(moving_rgba, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=[0,0,0,0])

        alpha = rot_rgba[..., 3:] / 255.0
        # Since rot_rgba[..., :3] is pre-multiplied, we just add it to the background
        comp = clean_bg * (1 - alpha) + rot_rgba[..., :3]
        comp = np.clip(comp, 0, 255).astype(np.uint8)

        writer.append_data(comp)
        
    writer.close()
    print("Video saved to /app/output/video.mp4")

if __name__ == "__main__":
    generate_video()
