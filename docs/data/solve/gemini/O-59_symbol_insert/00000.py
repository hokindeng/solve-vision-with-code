import cv2
import numpy as np
import imageio
import os

def ease_in_out(t):
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 3) / 2

def main():
    img = cv2.imread('/app/first_frame.png')
    
    sym5 = img[464:464+97, 622:622+97].copy()
    sym6 = img[464:464+97, 727:727+97].copy()

    # Extract new symbol from the reference panel
    panel = img[18:138, 887:1007]
    crop = panel[12:12+95, 12:12+95]
    new_sym = np.full((97, 97, 3), 170, dtype=np.uint8)
    new_sym[1:96, 1:96] = crop

    # Clear original sym5 and sym6 from base image
    base_img = img.copy()
    base_img[464:561, 622:824] = [255, 255, 255]

    frames = []

    for f in range(32):
        frame_img = base_img.copy()

        if f <= 12:
            t = f / 12.0
            progress = ease_in_out(t)
            x_sym5 = int(622 + 105 * progress)
            x_sym6 = int(727 + 105 * progress)
            alpha = 0.0
            y_new = 300
        elif f <= 18:
            t = (f - 12) / 6.0
            x_sym5 = 622 + 105
            x_sym6 = 727 + 105
            alpha = t
            y_new = 300
        else:
            t = (f - 18) / 13.0
            progress = ease_in_out(t)
            x_sym5 = 622 + 105
            x_sym6 = 727 + 105
            alpha = 1.0
            y_new = int(300 + (464 - 300) * progress)

        # Draw existing moving symbols
        frame_img[464:561, x_sym5:x_sym5+97] = sym5
        frame_img[464:561, x_sym6:x_sym6+97] = sym6
        
        # Draw new symbol fading/sliding
        if alpha > 0:
            if alpha < 1.0:
                roi = frame_img[y_new:y_new+97, 622:622+97]
                blended = cv2.addWeighted(new_sym, alpha, roi, 1 - alpha, 0)
                frame_img[y_new:y_new+97, 622:622+97] = blended
            else:
                frame_img[y_new:y_new+97, 622:622+97] = new_sym
                
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    # Write to video
    # H.264, yuv420p, 16fps
    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, macro_block_size=1, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    main()
