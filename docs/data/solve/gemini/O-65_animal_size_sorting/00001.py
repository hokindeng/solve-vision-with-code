import cv2
import numpy as np
from PIL import Image
import os
import subprocess

def ease_in_out_cubic(t):
    return 4 * t * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 3) / 2

def main():
    os.makedirs('/app/output', exist_ok=True)
    
    img = Image.open('/app/first_frame.png').convert('RGB')
    img_np = np.array(img)
    bg_color = img_np[0,0]
    
    mask = np.any(img_np != bg_color, axis=-1).astype(np.uint8) * 255
    kernel = np.ones((15, 15), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=2)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, 8, cv2.CV_32S)
    
    faces = []
    baseline_pixels = None
    baseline_y = 944
    
    # Extract background (just the bg_color + the baseline)
    bg_img = np.full_like(img_np, bg_color)
    
    for i in range(1, num_labels):
        comp_mask = (labels == i)
        orig_pixels_y, orig_pixels_x = np.where(comp_mask & (mask > 0))
        if len(orig_pixels_x) == 0:
            continue
            
        x_min, x_max = orig_pixels_x.min(), orig_pixels_x.max()
        y_min, y_max = orig_pixels_y.min(), orig_pixels_y.max()
        w = x_max - x_min + 1
        h = y_max - y_min + 1
        
        if w > 900: # Baseline
            baseline_y = y_min
            # copy baseline to bg_img
            bg_img[orig_pixels_y, orig_pixels_x] = img_np[orig_pixels_y, orig_pixels_x]
        else:
            # extract sprite
            sprite_mask = (comp_mask & (mask > 0))[y_min:y_max+1, x_min:x_max+1]
            sprite_rgb = img_np[y_min:y_max+1, x_min:x_max+1]
            
            # create RGBA sprite
            sprite_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            sprite_rgba[..., :3] = sprite_rgb
            sprite_rgba[..., 3] = sprite_mask * 255
            
            # use visible pixel count as area for precise sorting
            area = np.sum(sprite_mask)
            faces.append({
                'start_x': x_min,
                'start_y': y_min,
                'w': w,
                'h': h,
                'area': area,
                'sprite': sprite_rgba
            })
            
    # Sort faces by area, smallest to largest
    faces.sort(key=lambda f: f['area'])
    
    num_faces = len(faces)
    
    # Calculate final positions
    for i, f in enumerate(faces):
        f['final_x'] = int(1024 * (i + 1) / (num_faces + 1) - f['w'] / 2)
        f['final_y'] = int(baseline_y - f['h'])
        
    num_frames = 40
    fps = 16
    
    video_writer = cv2.VideoWriter('/app/output/temp_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (1024, 1024))
    
    for frame_idx in range(num_frames):
        # Calculate interpolation factor
        t = frame_idx / (num_frames - 1)
        t = max(0.0, min(1.0, t))
        eased_t = ease_in_out_cubic(t)
        
        # Start with clean background
        frame = bg_img.copy()
        
        # Draw faces
        for f in faces:
            curr_x = int(f['start_x'] + (f['final_x'] - f['start_x']) * eased_t)
            curr_y = int(f['start_y'] + (f['final_y'] - f['start_y']) * eased_t)
            
            # paste sprite
            sprite = f['sprite']
            h, w = sprite.shape[:2]
            
            mask_bool = (sprite[:, :, 3] > 0)
            for c in range(3):
                frame[curr_y:curr_y+h, curr_x:curr_x+w, c][mask_bool] = sprite[:, :, c][mask_bool]
                
        # BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
            
    video_writer.release()
    
    # Repackage to correct format using ffmpeg to ensure compatibility (yuv420p)
    subprocess.run([
        'ffmpeg', '-y', '-i', '/app/output/temp_video.mp4', 
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ], check=True)
    os.remove('/app/output/temp_video.mp4')

if __name__ == '__main__':
    main()
