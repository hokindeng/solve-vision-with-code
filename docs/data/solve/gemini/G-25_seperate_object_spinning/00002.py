import cv2
import numpy as np
import os
import math

def run():
    img = cv2.imread('/app/first_frame.png')
    
    # Extract background
    bg = img.copy()
    colors_to_remove = [
        [120, 120, 120],
        [100, 100, 200],
        [147, 20, 255],
        [180, 180, 180],
        [200, 150, 100],
        [255, 150, 100]
    ]
    for c in colors_to_remove:
        mask = cv2.inRange(img, np.array(c), np.array(c))
        bg[mask > 0] = [255, 255, 255]

    # Find objects using outline
    mask_objects = cv2.inRange(img, np.array([120, 120, 120]), np.array([120, 120, 120]))
    contours, _ = cv2.findContours(mask_objects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    objects = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cx = int(x + w / 2.0)
        cy = int(y + h / 2.0)
        
        area = cv2.contourArea(cnt)
        angle = 0.0
        if area < 5000: # Triangle
            angle = 14.0
        elif 9000 < area < 9300: # Small Square
            angle = -35.0
        elif area > 49000: # Large Square
            angle = -102.5
        elif 9300 <= area < 10000: # Hexagon
            angle = -0.5
        elif 40000 < area < 49000: # Octagon
            angle = -2.5
            
        size = int(math.hypot(w, h)) + 20
        if size % 2 != 0: size += 1
            
        patch = np.zeros((size, size, 4), dtype=np.uint8)
        
        sy, ey = max(0, cy - size//2), min(img.shape[0], cy + size//2)
        sx, ex = max(0, cx - size//2), min(img.shape[1], cx + size//2)
        
        dy1 = size//2 - (cy - sy)
        dy2 = size//2 + (ey - cy)
        dx1 = size//2 - (cx - sx)
        dx2 = size//2 + (ex - cx)
        
        obj_pixels = img[sy:ey, sx:ex]
        
        is_obj = np.any(obj_pixels != [255, 255, 255], axis=-1)
        obj_alpha = np.zeros(obj_pixels.shape[:2], dtype=np.uint8)
        obj_alpha[is_obj] = 255
        
        patch[dy1:dy2, dx1:dx2, :3] = obj_pixels
        patch[dy1:dy2, dx1:dx2, 3] = obj_alpha
        
        objects.append({
            'patch': patch,
            'cx': cx,
            'cy': cy,
            'target_angle': angle,
            'target_dx': 490.0
        })

    os.makedirs('/app/output', exist_ok=True)
    out_path = '/app/output/video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, 16.0, (1024, 1024))
    
    num_frames = 48
    for f in range(num_frames):
        t = f / (num_frames - 1)
        
        progress_rot = min(1.0, t * 2.0)
        progress_trans = max(0.0, (t - 0.5) * 2.0)
        
        frame = bg.copy()
        
        for obj in objects:
            curr_angle = obj['target_angle'] * progress_rot
            curr_dx = obj['target_dx'] * progress_trans
            
            patch = obj['patch']
            size = patch.shape[0]
            M = cv2.getRotationMatrix2D((size//2, size//2), curr_angle, 1.0)
            rotated_patch = cv2.warpAffine(patch, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
            
            curr_cx = obj['cx'] + int(curr_dx)
            curr_cy = obj['cy']
            
            sy, ey = max(0, curr_cy - size//2), min(frame.shape[0], curr_cy + size//2)
            sx, ex = max(0, curr_cx - size//2), min(frame.shape[1], curr_cx + size//2)
            
            dy1 = size//2 - (curr_cy - sy)
            dy2 = size//2 + (ey - curr_cy)
            dx1 = size//2 - (curr_cx - sx)
            dx2 = size//2 + (ex - curr_cx)
            
            if sy >= ey or sx >= ex: continue
            
            alpha = rotated_patch[dy1:dy2, dx1:dx2, 3] / 255.0
            
            bg_slice = frame[sy:ey, sx:ex]
            fg_slice = rotated_patch[dy1:dy2, dx1:dx2, :3]
            
            alpha_3d = alpha[:, :, np.newaxis]
            blended = np.round((1.0 - alpha_3d) * bg_slice + alpha_3d * fg_slice).astype(np.uint8)
            frame[sy:ey, sx:ex] = blended
                
        out.write(frame)
        
    out.release()
    
    os.system(f"ffmpeg -y -i {out_path} -vcodec libx264 -pix_fmt yuv420p /app/output/temp.mp4 > /dev/null 2>&1")
    os.system(f"mv /app/output/temp.mp4 {out_path}")

if __name__ == '__main__':
    run()
