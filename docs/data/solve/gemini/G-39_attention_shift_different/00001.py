import cv2
import numpy as np
import os
import subprocess
import shutil

def main():
    # Setup directories
    frames_dir = '/app/frames'
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)

    output_dir = '/app/output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load first frame
    img = cv2.imread('/app/first_frame.png')
    
    # Find the green box mask
    green_color = [70, 140, 70] # BGR
    mask = np.all(img == green_color, axis=-1)
    
    # Create clean image (background without the green box)
    # Background is white
    clean_img = img.copy()
    clean_img[mask] = [255, 255, 255]
    
    # Box original coordinates
    y, x = np.where(mask)
    
    # Target shift
    # DX = 474, DY = 1 precisely centers the box on the right object
    DX = 474
    DY = 1
    
    # Generate frames
    num_frames = 25
    for i in range(num_frames):
        t = i / (num_frames - 1)
        dx = int(round(DX * t))
        dy = int(round(DY * t))
        
        y_new = y + dy
        x_new = x + dx
        
        frame = clean_img.copy()
        frame[y_new, x_new] = green_color
        
        frame_path = os.path.join(frames_dir, f'frame_{i:04d}.png')
        cv2.imwrite(frame_path, frame)
        
    # Compile video using ffmpeg
    output_path = '/app/output/video.mp4'
    if os.path.exists(output_path):
        os.remove(output_path)
        
    ffmpeg_cmd = [
        'ffmpeg',
        '-y', # Overwrite
        '-framerate', '16',
        '-i', os.path.join(frames_dir, 'frame_%04d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    
    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Clean up frames
    shutil.rmtree(frames_dir)
    
    print(f"Video saved to {output_path}")

if __name__ == '__main__':
    main()
