import cv2
import numpy as np
import subprocess
import os

def generate_video():
    img = cv2.imread('/app/first_frame.png')
    h, w_img, _ = img.shape
    
    # Identify background and foreground
    diff = cv2.absdiff(img, np.array([255, 255, 255], dtype=np.uint8))
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    obj1_mask = np.zeros_like(mask)
    obj2_mask = np.zeros_like(mask)
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 1000:
            if stats[i, cv2.CC_STAT_WIDTH] < 120:
                obj1_mask[labels == i] = 255
            else:
                obj2_mask[labels == i] = 255

    # Target shifts based on bounding box analysis
    shift1_total = 303
    shift2_total = 477
    
    # Create clean background
    bg = img.copy()
    bg[obj1_mask == 255] = [255, 255, 255]
    bg[obj2_mask == 255] = [255, 255, 255]
    
    # Also extract the RGB pixels for the objects
    obj1_pixels = img.copy()
    obj1_pixels[obj1_mask == 0] = [0, 0, 0] 
    
    obj2_pixels = img.copy()
    obj2_pixels[obj2_mask == 0] = [0, 0, 0]
    
    num_frames = 30
    fps = 16
    
    out_path = '/app/output/video.mp4'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('/app/temp.mp4', fourcc, fps, (w_img, h))
    
    for i in range(num_frames):
        frame = bg.copy()
        
        # Calculate shifts
        s1 = int(round(shift1_total * i / (num_frames - 1)))
        s2 = int(round(shift2_total * i / (num_frames - 1)))
        
        shifted_mask1 = np.roll(obj1_mask, s1, axis=1)
        shifted_pixels1 = np.roll(obj1_pixels, s1, axis=1)
        
        shifted_mask2 = np.roll(obj2_mask, s2, axis=1)
        shifted_pixels2 = np.roll(obj2_pixels, s2, axis=1)
        
        # Paste Object 1
        frame[shifted_mask1 == 255] = shifted_pixels1[shifted_mask1 == 255]
        
        # Paste Object 2
        frame[shifted_mask2 == 255] = shifted_pixels2[shifted_mask2 == 255]
        
        out.write(frame)
        
    out.release()
    
    # Re-encode to H.264
    subprocess.run([
        'ffmpeg', '-y', '-i', '/app/temp.mp4',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        out_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists('/app/temp.mp4'):
        os.remove('/app/temp.mp4')

if __name__ == '__main__':
    generate_video()
