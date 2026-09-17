import cv2
import numpy as np
import os
import subprocess

def main():
    img = cv2.imread('/app/first_frame.png')
    h, w = img.shape[:2]
    
    # 1. Isolate the objects
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(img.copy(), mask, (0, 0), (0, 0, 0), loDiff=(0,0,0), upDiff=(0,0,0))
    bg_mask = mask[1:h+1, 1:w+1]
    non_bg = cv2.bitwise_not(bg_mask * 255)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(non_bg, connectivity=8)
    
    objects = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 1000:
            objects.append(i)
            
    base_bg = img.copy()
    for i in objects:
        base_bg[labels == i] = [255, 255, 255]
        
    os.makedirs('/app/output', exist_ok=True)
    frames_dir = '/app/output/frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    # 2. Generate frames
    num_frames = 17
    max_angle = 64.0
    
    for frame_idx in range(num_frames):
        if frame_idx == 0:
            frame = img.copy()
        else:
            frame = base_bg.copy()
            angle = (frame_idx / (num_frames - 1)) * max_angle
            
            for i in objects:
                cx, cy = centroids[i]
                # In OpenCV, positive angle means counter-clockwise
                M_rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
                
                obj_mask = (labels == i).astype(np.uint8)
                dilated_mask = cv2.dilate(obj_mask, np.ones((5,5), np.uint8))
                
                O_i = img.copy()
                O_i[dilated_mask == 0] = [255, 255, 255]
                
                O_rot = cv2.warpAffine(O_i, M_rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
                M_rot_soft = cv2.warpAffine(dilated_mask * 255, M_rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) / 255.0
                M_rot_soft = M_rot_soft[:, :, np.newaxis]
                
                frame = O_rot * M_rot_soft + frame * (1 - M_rot_soft)
                
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            
        cv2.imwrite(f'{frames_dir}/frame_{frame_idx:04d}.png', frame)
        
    # 3. Create video using ffmpeg
    cmd = [
        'ffmpeg', '-y', '-framerate', '16', '-i', f'{frames_dir}/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ]
    subprocess.run(cmd, check=True)
    
if __name__ == '__main__':
    main()
