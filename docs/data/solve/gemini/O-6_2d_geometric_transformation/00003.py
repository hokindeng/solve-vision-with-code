import cv2
import numpy as np
import os
import subprocess

def create_video():
    img = cv2.imread('/app/first_frame.png')
    h, w = img.shape[:2]
    
    # --- 1. Compute static_bg ---
    static_bg = img.copy()
    
    # Find moving part
    colored_mask = cv2.inRange(img, np.array([50, 140, 180]), np.array([60, 160, 200]))
    moving_mask = np.zeros((h, w), dtype=np.uint8)
    contours, _ = cv2.findContours(colored_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            cv2.drawContours(moving_mask, [cnt], -1, 255, -1)
            cv2.drawContours(moving_mask, [cnt], -1, 255, 6) # Expand to clear anti-aliasing
            
    # Erase moving part
    static_bg[moving_mask == 255] = [240, 240, 240]
    
    # Restore the perfect center marker on static_bg
    # We use the left half of the original marker, which is uncovered, and mirror it
    left_side = img[455:490, 220:239].copy()
    right_side = cv2.flip(left_side, 1)
    center_col = img[455:490, 239:240].copy()
    reconstructed_marker = np.hstack([left_side, center_col, right_side])
    
    # Only copy pixels within marker radius to avoid copying dashed outline
    Y, X = np.ogrid[:35, :39]
    dist = np.sqrt((X - 19)**2 + (Y - 17)**2)
    marker_mask = dist <= 12.5
    
    roi = static_bg[455:490, 220:259]
    roi[marker_mask] = reconstructed_marker[marker_mask]

    # --- 2. Compute sprite ---
    # Sprite moving mask must also include the center marker so it rotates nicely
    cv2.circle(moving_mask, (239, 472), 14, 255, -1)
    
    alpha = 1.0 - np.min(img, axis=2).astype(np.float32) / 240.0
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha[colored_mask > 0] = 1.0
    alpha[moving_mask == 0] = 0.0
    
    alpha_uint8 = (alpha * 255).astype(np.uint8)
    
    sprite = np.zeros((h, w, 4), dtype=np.uint8)
    sprite[:, :, :3] = img
    sprite[:, :, 3] = alpha_uint8
    
    # --- 3. Generate Video ---
    os.makedirs('/app/output', exist_ok=True)
    out_path = '/app/output/video.mp4'
    
    fps = 16
    total_frames = 70
    start_pause = 5
    end_pause = 5
    rot_frames = total_frames - start_pause - end_pause
    
    center = (239, 472)
    max_angle = 143.0
    
    process = subprocess.Popen([
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}', '-pix_fmt', 'bgr24', '-r', str(fps),
        '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out_path
    ], stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    
    bg_float = static_bg.astype(np.float32)
    
    for i in range(total_frames):
        if i < start_pause:
            progress = 0.0
        elif i >= total_frames - end_pause:
            progress = 1.0
        else:
            # smoothstep
            t = (i - start_pause) / (rot_frames - 1)
            progress = (1.0 - np.cos(t * np.pi)) / 2.0
            
        angle = max_angle * progress
        
        # Exact original frame for the very first frame
        if i == 0:
            frame = img.copy()
        else:
            M = cv2.getRotationMatrix2D(center, -angle, 1.0)
            rotated_sprite = cv2.warpAffine(sprite, M, (w, h))
            
            fg = rotated_sprite[:, :, :3].astype(np.float32)
            alpha_channel = rotated_sprite[:, :, 3].astype(np.float32) / 255.0
            alpha_3 = np.dstack((alpha_channel, alpha_channel, alpha_channel))
            
            frame = (fg * alpha_3 + bg_float * (1.0 - alpha_3)).astype(np.uint8)
            
        process.stdin.write(frame.tobytes())
        
    process.stdin.close()
    process.wait()

if __name__ == '__main__':
    create_video()
