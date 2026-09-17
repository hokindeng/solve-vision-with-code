import cv2
import numpy as np
import imageio
import os

def main():
    img1 = cv2.imread('/app/first_frame.png')
    h, w, _ = img1.shape
    # img1 is BGR. The color in img1 is B=153, G=53, R=119.
    color = (153, 53, 119) # BGR

    clean_img = img1.copy()
    # Remove question marks
    clean_img[650:715, 450:510] = 255
    clean_img[650:715, 750:820] = 255

    img_step1 = clean_img.copy()
    # Intermediate shape
    cv2.rectangle(img_step1, (414, 666), (546, 698), color, 6, cv2.LINE_AA)

    img_step2 = img_step1.copy()
    # Final shape
    cv2.rectangle(img_step2, (721, 666), (853, 698), color, 4, cv2.LINE_AA)

    frames = []
    for i in range(16):
        if i < 8:
            alpha = i / 7.0
            frame = cv2.addWeighted(img1, 1 - alpha, img_step1, alpha, 0)
        else:
            alpha = (i - 8) / 7.0
            frame = cv2.addWeighted(img_step1, 1 - alpha, img_step2, alpha, 0)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, macro_block_size=None, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    main()
