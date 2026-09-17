import cv2
import numpy as np
from PIL import Image
import os
import subprocess

def main():
    os.makedirs('/app/output/frames', exist_ok=True)
    
    # Load original first frame
    img_cv = cv2.imread('/app/first_frame.png')
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    # The uniform background color
    bg_color = np.array([240, 240, 240])
    
    # Bounding boxes for each domino (x, y, w, h)
    # Extracted exactly from first_frame.png contours
    rects = [
        (59, 456, 69, 127),   # 0: Green (START)
        (211, 456, 53, 127),  # 1: Gray trunk
        (355, 363, 53, 127),  # 2: Red Branch A - 1
        (355, 549, 53, 127),  # 3: Cyan Branch B - 1
        (499, 338, 53, 127),  # 4: Red Branch A - 2
        (499, 573, 53, 127),  # 5: Cyan Branch B - 2
        (643, 313, 53, 127),  # 6: Red Branch A - 3
        (643, 598, 53, 127)   # 7: Cyan Branch B - 3
    ]
    
    base_bg_np = img_cv.copy()
    domino_imgs = []
    
    # Extract each domino and remove it from the base background
    for i, (x, y, w, h) in enumerate(rects):
        roi = base_bg_np[y:y+h, x:x+w]
        
        # Pixels that differ from the background color are considered part of the domino
        alpha_mask = np.any(np.abs(roi.astype(int) - bg_color) > 5, axis=-1)
        
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = roi
        rgba[alpha_mask, 3] = 255
        
        # Erase domino footprint from the background image
        base_bg_np[y:y+h, x:x+w][alpha_mask] = bg_color
        
        # Create a transparent canvas of the full image size to hold the domino
        # We fill transparent pixels with bg_color to prevent black fringes on rotation interpolation
        canvas = Image.new("RGBA", (img_cv.shape[1], img_cv.shape[0]), (240, 240, 240, 0))
        dom_img = Image.fromarray(rgba)
        canvas.paste(dom_img, (x, y))
        
        domino_imgs.append({
            'canvas': canvas,
            'pivot': (x + w, y + h) # Domino rotates around its bottom-right corner
        })
        
    base_bg = Image.fromarray(base_bg_np).convert("RGBA")
    
    # Animation parameters
    omega = 6.0       # Rotation speed (degrees per frame)
    max_angle = 70.0  # Final resting angle for all dominos
    
    # Calculate start frame for each domino using collision angles
    start_frames = [0.0] * 8
    
    # Physics timings (calculated to match 16fps over ~50 frames)
    start_frames[0] = 0.0
    start_frames[1] = start_frames[0] + 41.0 / omega
    start_frames[2] = start_frames[1] + 57.0 / omega
    start_frames[3] = start_frames[1] + 68.0 / omega
    start_frames[4] = start_frames[2] + 46.0 / omega
    start_frames[6] = start_frames[4] + 46.0 / omega
    start_frames[5] = start_frames[3] + 46.0 / omega
    start_frames[7] = start_frames[5] + 46.0 / omega
    
    total_frames = 50
    
    # Render frames
    for f in range(total_frames):
        frame_img = base_bg.copy()
        
        for i in range(8):
            sf = start_frames[i]
            if f < sf:
                angle = 0.0
            else:
                angle = (f - sf) * omega
                if angle > max_angle:
                    angle = max_angle
            
            # Rotate canvas counter-clockwise by -angle (which means clockwise falling right)
            rotated = domino_imgs[i]['canvas'].rotate(-angle, resample=Image.BICUBIC, center=domino_imgs[i]['pivot'])
            
            # Composite the rotated domino over the background
            frame_img = Image.alpha_composite(frame_img, rotated)
            
        frame_rgb = frame_img.convert("RGB")
        frame_rgb.save(f'/app/output/frames/frame_{f:04d}.png')
        
    # Generate MP4 using ffmpeg
    subprocess.run([
        'ffmpeg', '-y', '-framerate', '16', '-i', '/app/output/frames/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ], check=True)

if __name__ == '__main__':
    main()
