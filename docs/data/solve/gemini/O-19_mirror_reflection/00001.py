import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    base_img = img.copy()

    # 1. Remove text
    for y in range(400, 462):
        for x in range(700, 900):
            base_img[y, x] = [255, 255, 255]

    # 2. Keep only incident ray (and arrow head) for y in 400 to 462, x in 600 to 700
    for y in range(400, 462):
        for x in range(600, 700):
            if np.all(img[y, x] == [0, 0, 0]):
                continue
            if not np.all(img[y, x] == [255, 255, 255]):
                x_ideal = 442 + (y - 49) * 242 / 416
                dev = abs(x - x_ideal)
                if dev > 4.5:
                    base_img[y, x] = [255, 255, 255]

    # 3. Remove normal line over and below the mirror
    for y in range(462, 550):
        if np.all(img[y, 685] == [150, 150, 150]):
            left = img[y, 684].astype(np.float32)
            right = img[y, 686].astype(np.float32)
            if not np.all(left == 255) and not np.all(right == 255):
                base_img[y, 685] = ((left + right) / 2).astype(np.uint8)
            elif not np.all(left == 255):
                base_img[y, 685] = left.astype(np.uint8)
            elif not np.all(right == 255):
                base_img[y, 685] = right.astype(np.uint8)
            else:
                base_img[y, 685] = [255, 255, 255]

    frames = []
    # Frame 0 is first_frame.png
    frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    TOTAL_FRAMES = 35
    for i in range(1, TOTAL_FRAMES):
        frame_img = base_img.copy()
        
        p = i / float(TOTAL_FRAMES - 1)
        x_start, y_start = 684, 465
        x_end, y_end = 955, 0
        
        x_cur = int(x_start + (x_end - x_start) * p)
        y_cur = int(y_start + (y_end - y_start) * p)
        
        if x_cur != x_start or y_cur != y_start:
            cv2.line(frame_img, (x_start, y_start), (x_cur, y_cur), (117, 0, 0), 2, lineType=cv2.LINE_8)
            
        frames.append(cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB))

    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    solve()
