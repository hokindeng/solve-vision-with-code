import cv2
import numpy as np
import math
import subprocess
import os

def paste_patch(frame, patch_bgr, patch_mask, x1, y1):
    h, w = frame.shape[:2]
    ph, pw = patch_bgr.shape[:2]
    
    fx1, fy1 = int(round(x1)), int(round(y1))
    fx2, fy2 = fx1 + pw, fy1 + ph
    
    px1, py1 = 0, 0
    px2, py2 = pw, ph
    
    if fx1 < 0:
        px1 -= fx1
        fx1 = 0
    if fy1 < 0:
        py1 -= fy1
        fy1 = 0
    if fx2 > w:
        px2 -= (fx2 - w)
        fx2 = w
    if fy2 > h:
        py2 -= (fy2 - h)
        fy2 = h
        
    if px1 >= px2 or py1 >= py2:
        return
        
    roi = frame[fy1:fy2, fx1:fx2]
    m = patch_mask[py1:py2, px1:px2]
    bgr = patch_bgr[py1:py2, px1:px2]
    
    idx = m > 128
    roi[idx] = bgr[idx]
    frame[fy1:fy2, fx1:fx2] = roi

def main():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("Cannot read first_frame.png")
        
    bg_color = img[0, 0].tolist()
    h, w = img.shape[:2]

    left_half = img[:, :w//2]
    mask_objects = cv2.inRange(left_half, np.array(bg_color), np.array(bg_color))
    mask_objects = cv2.bitwise_not(mask_objects)
    contours_left, _ = cv2.findContours(mask_objects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bg_img = img.copy()
    left_bg = bg_img[:, :w//2]
    left_bg[mask_objects > 0] = bg_color

    target_params = [
        {'cx': 170, 'cy': 625, 'angle': -12, 'dx': 633},
        {'cx': 371, 'cy': 625, 'angle': -8, 'dx': 305},
        {'cx': 391, 'cy': 432, 'angle': -29, 'dx': 459},
        {'cx': 172, 'cy': 433, 'angle': 0, 'dx': 504}
    ]

    objects = []
    for c in contours_left:
        M = cv2.moments(c)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        
        best_param = None
        min_dist = 99999
        for p in target_params:
            dist = math.hypot(cx - p['cx'], cy - p['cy'])
            if dist < min_dist:
                min_dist = dist
                best_param = p
                
        obj_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(obj_mask, [c], -1, 255, -1)
        
        x, y, wc, hc = cv2.boundingRect(c)
        pad = int(math.hypot(wc, hc) / 2) + 20
        x1, y1 = max(0, cx - pad), max(0, cy - pad)
        x2, y2 = min(w, cx + pad), min(h, cy + pad)
        
        obj_bgr_patch = img[y1:y2, x1:x2].copy()
        obj_mask_patch = obj_mask[y1:y2, x1:x2].copy()
        
        objects.append({
            'cx': cx, 'cy': cy,
            'bgr_patch': obj_bgr_patch,
            'mask_patch': obj_mask_patch,
            'x1': x1, 'y1': y1,
            'angle': best_param['angle'],
            'dx': best_param['dx']
        })

    os.makedirs('/app/output', exist_ok=True)
    frames_dir = '/app/output/frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    total_frames = 48
    fps = 16
    
    phase1_frames = 24
    phase2_frames = total_frames - phase1_frames
    
    for frame in range(total_frames):
        if frame < phase1_frames:
            t_rot = frame / (phase1_frames - 1)
            t_trans = 0.0
        else:
            t_rot = 1.0
            t_trans = (frame - phase1_frames + 1) / phase2_frames
            
        frame_img = bg_img.copy()
        for obj in objects:
            cur_angle = obj['angle'] * t_rot
            cur_dx = obj['dx'] * t_trans
            
            patch_h, patch_w = obj['bgr_patch'].shape[:2]
            center = (obj['cx'] - obj['x1'], obj['cy'] - obj['y1'])
            
            rot_mat = cv2.getRotationMatrix2D(center, cur_angle, 1.0)
            warped_bgr = cv2.warpAffine(obj['bgr_patch'], rot_mat, (patch_w, patch_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)
            warped_mask = cv2.warpAffine(obj['mask_patch'], rot_mat, (patch_w, patch_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            paste_patch(frame_img, warped_bgr, warped_mask, obj['x1'] + cur_dx, obj['y1'])

        cv2.imwrite(f'{frames_dir}/frame_{frame:03d}.png', frame_img)
        
    cmd = [
        'ffmpeg', '-y', '-framerate', str(fps),
        '-i', f'{frames_dir}/frame_%03d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '/app/output/video.mp4'
    ]
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
