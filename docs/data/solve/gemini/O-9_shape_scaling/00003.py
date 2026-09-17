import cv2
import numpy as np
import os
import imageio

def generate_video():
    img = cv2.imread('/app/first_frame.png')
    # C's bounding box
    C = img[693:844, 180:331]
    
    # The area containing the '?' (including its dot)
    box_orig = img[730:800, 730:800].copy()

    frames = []

    for i in range(60):
        canvas = img.copy()
        
        # Fade out '?' over frames 6 to 15
        if 6 <= i <= 15:
            t = (i - 5) / 10.0
            blended_box = box_orig * (1 - t) + 255 * t
            canvas[730:800, 730:800] = blended_box.astype(np.uint8)
        elif i > 15:
            # Keep it fully erased
            canvas[730:800, 730:800] = 255
            
        # Move and scale C from frame 16 to 50
        if i >= 16:
            if i <= 50:
                t = (i - 15) / 35.0
                t_ease = t * t * (3 - 2 * t)
            else:
                t_ease = 1.0
                
            # size interpolates from 151 to 127
            size = int(round(151 * (1 - t_ease) + 127 * t_ease))
            # x center interpolates from 255.5 to 769.5
            cx = 255.5 + t_ease * (769.5 - 255.5)
            # y center is constant at 768.5
            cy = 768.5
            
            # Use INTER_NEAREST to preserve aliased 3-color look
            resized_C = cv2.resize(C, (size, size), interpolation=cv2.INTER_NEAREST)
            left = int(round(cx - size / 2.0))
            top = int(round(cy - size / 2.0))
            
            # Mask out the white background (black and pink pixels make up the shape)
            mask = (resized_C != 255).any(axis=2)
            roi = canvas[top:top+size, left:left+size]
            
            # Draw the shape onto the canvas
            roi[mask] = resized_C[mask]

        # Convert BGR to RGB for imageio
        frames.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    os.makedirs('/app/output', exist_ok=True)
    # Write video
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    generate_video()
