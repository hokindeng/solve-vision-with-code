import cv2
import numpy as np
import math
import os

def create_video():
    img = cv2.imread('/app/first_frame.png')
    cx, cy = 512, 512
    bg_color = np.array([255, 248, 240], dtype=np.uint8)

    # 1. Isolate the hands component
    mask = (img != bg_color).any(axis=2).astype(np.uint8) * 255
    y, x = np.ogrid[:1024, :1024]
    circle_mask = (x - cx)**2 + (y - cy)**2 <= 300**2
    inside_mask = mask & circle_mask

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inside_mask, connectivity=8)
    label = labels[512, 512]
    comp_mask = (labels == label)

    # 2. Separate into pin, minute hand, and hour hand
    is_pin = comp_mask & (img[:,:,0] < 50) & (img[:,:,1] < 50) & (img[:,:,2] < 50)
    hands_mask = comp_mask & ~is_pin

    pts = np.column_stack(np.where(hands_mask))

    min_angle = 216
    hr_angle = 228

    rad_min = math.radians(min_angle - 90)
    vec_min = np.array([math.cos(rad_min), math.sin(rad_min)])

    rad_hr = math.radians(hr_angle - 90)
    vec_hr = np.array([math.cos(rad_hr), math.sin(rad_hr)])

    is_min = np.zeros_like(hands_mask, dtype=bool)
    is_hr = np.zeros_like(hands_mask, dtype=bool)

    for py, px in pts:
        v = np.array([px - cx, py - cy])
        cross_min = vec_min[0]*v[1] - vec_min[1]*v[0]
        cross_hr = vec_hr[0]*v[1] - vec_hr[1]*v[0]
        
        dist_min = abs(cross_min)
        dist_hr = abs(cross_hr)
        
        r = math.hypot(px - cx, py - cy)
        if r > 215:
            is_min[py, px] = True
        else:
            if dist_min < dist_hr:
                is_min[py, px] = True
            else:
                is_hr[py, px] = True

    # 3. Create RGBA layers padded with background color
    min_rgba = np.zeros((1024, 1024, 4), dtype=np.uint8)
    min_rgba[:, :, :3] = bg_color
    min_rgba[is_min, :3] = img[is_min]
    min_rgba[is_min, 3] = 255

    hr_rgba = np.zeros((1024, 1024, 4), dtype=np.uint8)
    hr_rgba[:, :, :3] = bg_color
    hr_rgba[is_hr, :3] = img[is_hr]
    hr_rgba[is_hr, 3] = 255

    pin_rgba = np.zeros((1024, 1024, 4), dtype=np.uint8)
    pin_rgba[:, :, :3] = bg_color
    pin_rgba[is_pin, :3] = img[is_pin]
    pin_rgba[is_pin, 3] = 255

    clean_bg = img.copy()
    clean_bg[comp_mask] = bg_color

    # 4. Generate video
    os.makedirs('/app/output', exist_ok=True)
    out_path = '/app/output/video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # We will write frames to a temp directory and use ffmpeg to combine them 
    # to ensure precise encoding settings (H.264, yuv420p, etc as requested).
    os.makedirs('/app/output/frames', exist_ok=True)

    num_frames = 120
    for i in range(num_frames):
        # Angle offsets for frame i
        # Clockwise is negative in cv2.getRotationMatrix2D
        angle_min = -5760.0 * i / (num_frames - 1)
        angle_hr = -480.0 * i / (num_frames - 1)
        
        M_min = cv2.getRotationMatrix2D((cx, cy), angle_min, 1.0)
        min_rot = cv2.warpAffine(min_rgba, M_min, (1024, 1024), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 248, 240, 0))
        
        M_hr = cv2.getRotationMatrix2D((cx, cy), angle_hr, 1.0)
        hr_rot = cv2.warpAffine(hr_rgba, M_hr, (1024, 1024), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 248, 240, 0))
        
        res = clean_bg.copy()
        
        # Blend hour hand
        alpha_hr = hr_rot[:, :, 3] / 255.0
        for c in range(3):
            res[:, :, c] = res[:, :, c] * (1 - alpha_hr) + hr_rot[:, :, c] * alpha_hr
            
        # Blend minute hand
        alpha_min = min_rot[:, :, 3] / 255.0
        for c in range(3):
            res[:, :, c] = res[:, :, c] * (1 - alpha_min) + min_rot[:, :, c] * alpha_min
            
        # Blend center pin
        alpha_pin = pin_rgba[:, :, 3] / 255.0
        for c in range(3):
            res[:, :, c] = res[:, :, c] * (1 - alpha_pin) + pin_rgba[:, :, c] * alpha_pin
            
        cv2.imwrite(f'/app/output/frames/frame_{i:03d}.png', res)

    # Use ffmpeg to encode
    cmd = "ffmpeg -y -framerate 16 -i /app/output/frames/frame_%03d.png -c:v libx264 -pix_fmt yuv420p /app/output/video.mp4"
    os.system(cmd)
    
if __name__ == "__main__":
    create_video()
