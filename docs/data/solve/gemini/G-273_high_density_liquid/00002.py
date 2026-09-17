import cv2
import numpy as np
import imageio
import os

def ease(t):
    return t * t * (3 - 2 * t)

def main():
    img = cv2.imread('/app/first_frame.png')
    
    # Extract objects (they are 57x57)
    obj1 = img[94:151, 240:297].copy()
    obj2 = img[94:151, 484:541].copy()
    obj3 = img[94:151, 728:785].copy()
    
    # Create clean background
    bg = img.copy()
    bg[94:151, 240:297] = [255, 255, 255]
    bg[94:151, 484:541] = [255, 255, 255]
    bg[94:151, 728:785] = [255, 255, 255]
    
    frames = []
    num_frames = 80
    
    for i in range(num_frames):
        t = i / (num_frames - 1)
        e = ease(t)
        
        # Object 1 (Left): sinks to the bottom (max y is 787)
        y1 = int(94 + (787 - 94) * e)
        # Object 2 (Middle): sinks to the bottom (max y is 787)
        y2 = int(94 + (787 - 94) * e)
        # Object 3 (Right): floats in darker liquid (surface at 590, floats half-submerged at 562)
        y3 = int(94 + (562 - 94) * e)
        
        frame = bg.copy()
        
        frame[y1:y1+57, 240:297] = obj1
        frame[y2:y2+57, 484:541] = obj2
        frame[y3:y3+57, 728:785] = obj3
        
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', macro_block_size=None, pixelformat='yuv420p')
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == '__main__':
    main()
