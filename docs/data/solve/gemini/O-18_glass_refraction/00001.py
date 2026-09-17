import cv2
import numpy as np
import math
import os
import subprocess

def main():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("Could not read /app/first_frame.png")

    height, width, _ = img.shape

    # Generate a 'clean' version of the image where the annotation is removed.
    # The annotation is in the region x in [470, 650] and y in [450, 510].
    clean = img.copy()
    roi = clean[450:511, 470:650]
    
    # We want to remove the black/grey text and arc.
    # Identify non-white pixels
    non_white = (roi[:, :, 0] < 250) | (roi[:, :, 1] < 250) | (roi[:, :, 2] < 250)
    # Identify blueish pixels (the incident ray)
    blueish = (roi[:, :, 0] > roi[:, :, 1] + 50) & (roi[:, :, 0] > roi[:, :, 2] + 50)
    
    # Pixels to erase are non-white and not blueish
    to_remove = non_white & ~blueish
    roi[to_remove] = [255, 255, 255]
    clean[450:511, 470:650] = roi

    # Physical parameters
    n_air = 1.00
    n_glass = 1.387
    theta1_deg = 41.7
    theta1_rad = math.radians(theta1_deg)
    
    # Snell's law: n1 * sin(theta1) = n2 * sin(theta2)
    sin_theta2 = (n_air / n_glass) * math.sin(theta1_rad)
    theta2_rad = math.asin(sin_theta2)
    
    # The incident ray hits the interface at x=512, y=512
    x0, y0 = 512, 512
    
    # Refracted ray endpoints (extends to the image boundary)
    # The ray goes downwards (positive y) and rightwards (positive x).
    # Normal is vertical, so theta2 is the angle with the vertical.
    # dx = dy * tan(theta2)
    dy = (height - 1) - y0
    dx = dy * math.tan(theta2_rad)
    
    x1 = x0 + dx
    y1 = height - 1

    os.makedirs('/tmp/frames', exist_ok=True)

    num_frames = 70
    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 1.0
        
        # Blend the original image and the clean image to fade out the annotation
        frame = cv2.addWeighted(img, 1.0 - t, clean, t, 0)
        
        # Calculate current end point of the refracted ray
        cur_x = x0 + t * (x1 - x0)
        cur_y = y0 + t * (y1 - y0)
        
        # Draw the refracted ray (red)
        if i > 0:
            cv2.line(frame, (int(x0), int(y0)), (int(round(cur_x)), int(round(cur_y))), (0, 0, 255), 4, cv2.LINE_AA)
            
        frame_path = f'/tmp/frames/frame_{i:04d}.png'
        cv2.imwrite(frame_path, frame)

    # Compile frames to video
    os.makedirs('/app/output', exist_ok=True)
    out_video = '/app/output/video.mp4'
    if os.path.exists(out_video):
        os.remove(out_video)
        
    cmd = [
        'ffmpeg', '-y', '-framerate', '16', '-i', '/tmp/frames/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out_video
    ]
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
