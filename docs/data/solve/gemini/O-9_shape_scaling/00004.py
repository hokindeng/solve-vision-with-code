import cv2
import numpy as np
import math
import imageio
import os

def main():
    os.makedirs('/app/output', exist_ok=True)

    img_orig = cv2.imread('/app/first_frame.png')
    
    # Extract shape C
    # C is bounded by [693:844, 180:331]
    C = img_orig[693:844, 180:331].copy()

    frames = []
    
    # Frame 0 is perfectly identical to first_frame.png
    frames.append(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))

    target_scale = math.sqrt(3)/2

    for i in range(1, 60):
        img = img_orig.copy()
        
        # Erase the question mark area with white
        img[693:844, 694:845] = 255
        
        # Calculate scale progression (1.0 down to target_scale)
        progress = i / 59.0
        scale = 1.0 - (1.0 - target_scale) * progress
        
        # We warp C around its center (75.5, 75.5) and put it into the patch
        # The center of the patch is exactly the center we want to align to
        M = cv2.getRotationMatrix2D((75.5, 75.5), 0, scale)
        C_warped = cv2.warpAffine(
            C, 
            M, 
            (151, 151), 
            flags=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=(255, 255, 255)
        )
        
        # Paste it into the image
        img[693:844, 694:845] = C_warped
        
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Save video
    imageio.mimwrite(
        '/app/output/video.mp4', 
        frames, 
        fps=16, 
        macro_block_size=None, 
        quality=10, 
        codec='libx264', 
        pixelformat='yuv420p'
    )

if __name__ == "__main__":
    main()
