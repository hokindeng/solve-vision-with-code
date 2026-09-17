import cv2
import numpy as np
import imageio
import os

def get_pos(f, keyframes):
    for i in range(len(keyframes) - 1):
        f1, (x1, y1) = keyframes[i]
        f2, (x2, y2) = keyframes[i+1]
        if f1 <= f <= f2:
            if f1 == f2: return x1, y1
            t = (f - f1) / (f2 - f1)
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return int(round(x)), int(round(y))
    return keyframes[-1][1]

def main():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("first_frame.png not found")
        
    book1 = img[308:513, 736:768].copy()
    book2 = img[278:513, 773:805].copy()

    bg = img.copy()

    # Erase Book 1
    bg[308:512, 736:768] = [255, 255, 255]
    bg[512, 736:768] = [22, 41, 52]

    # Erase Book 2
    bg[278:512, 773:805] = [255, 255, 255]
    bg[512, 773:805] = [22, 41, 52]
    
    # Book 1 goes to (334, 308)
    kf1 = [(0, (736, 308)), (3, (736, 58)), (8, (334, 58)), (11, (334, 308)), (23, (334, 308))]
    
    # Book 2 goes to (556, 278)
    kf2 = [(0, (773, 278)), (11, (773, 278)), (14, (773, 28)), (19, (556, 28)), (22, (556, 278)), (23, (556, 278))]
    
    frames = []
    for f in range(24):
        frame = bg.copy()
        
        # We should draw them such that if they overlap, the one in motion is on top
        # Actually since they move sequentially, drawing 1 then 2 or 2 then 1 is fine.
        x1, y1 = get_pos(f, kf1)
        h1, w1 = book1.shape[:2]
        frame[y1:y1+h1, x1:x1+w1] = book1
        
        x2, y2 = get_pos(f, kf2)
        h2, w2 = book2.shape[:2]
        frame[y2:y2+h2, x2:x2+w2] = book2
        
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    os.makedirs('/app/output', exist_ok=True)
    
    # Write video
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == '__main__':
    main()
