import cv2
import numpy as np
import math
import imageio

def clean_image(img):
    clean_img = img.copy()
    # Remove angle annotation from top right quadrant
    for y in range(400, 510):
        for x in range(480, 700):
            b, g, r = [int(v) for v in img[y, x]]
            if b < 240 and b == g and g == r:
                clean_img[y, x] = [255, 255, 255]
    return clean_img

def main():
    img = cv2.imread('/app/first_frame.png')
    base_img = clean_image(img)

    num_frames = 70
    fps = 16

    x1, y1 = 512.1387, 512.0
    
    # Snell's Law
    n1 = 1.00
    n2 = 1.857
    theta1 = 56.3
    theta1_rad = math.radians(theta1)
    sin_theta2 = (n1 / n2) * math.sin(theta1_rad)
    theta2_rad = math.asin(sin_theta2)
    
    # End point
    dy = 1024 - y1
    dx = dy * math.tan(theta2_rad)
    x2 = x1 + dx
    y2 = 1024.0

    frames = []

    for i in range(num_frames):
        frame = base_img.copy()
        
        # Calculate current end point based on frame
        # at i=0, length is 0 (point). at i=69, length is full.
        progress = i / (num_frames - 1)
        cur_x = x1 + (x2 - x1) * progress
        cur_y = y1 + (y2 - y1) * progress

        # Draw line if it has some length
        if i > 0:
            cv2.line(
                frame, 
                (int(x1 * 16), int(y1 * 16)), 
                (int(cur_x * 16), int(cur_y * 16)), 
                (0, 0, 255), 
                3, 
                cv2.LINE_AA, 
                shift=4
            )
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    # Save video
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, codec='libx264', macro_block_size=None, pixelformat='yuv420p')
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == "__main__":
    main()
