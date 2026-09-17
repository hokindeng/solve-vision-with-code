import cv2
import numpy as np
import imageio
import os

def main():
    img_path = '/app/first_frame.png'
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {img_path}")
        
    bg_color = img[0, 0]
    mask = np.all(img != bg_color, axis=-1).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pentagon_cnt = None
    center = (0, 0)
    target_radius = 0
    
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 5:
            pentagon_cnt = cnt
            (x, y), r = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            target_radius = int(r) + 15  # a bit larger to encircle it nicely
            break
            
    if pentagon_cnt is None:
        raise ValueError("No pentagon found in the image")
        
    num_frames = 30
    fps = 16
    
    out_dir = '/app/output'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'video.mp4')
    
    writer = imageio.get_writer(out_path, fps=fps, codec='libx264', macro_block_size=None, quality=8, pixelformat='yuv420p')
    
    for i in range(num_frames):
        frame = img.copy()
        
        if i > 0:
            current_radius = int((i / (num_frames - 1)) * target_radius)
            if current_radius > 0:
                cv2.circle(frame, center, current_radius, (0, 0, 255), 6, cv2.LINE_AA)
            
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    main()
