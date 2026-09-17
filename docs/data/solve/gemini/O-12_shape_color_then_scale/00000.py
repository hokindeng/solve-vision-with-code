import cv2
import numpy as np
from PIL import Image, ImageDraw
import imageio

def generate_video():
    first_frame = cv2.imread('/app/first_frame.png')
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    
    # Colors (RGB)
    c1 = (114, 191, 38)   # color_155
    c2 = (114, 229, 210)  # color_246
    
    frames = []
    num_frames = 60
    
    for i in range(num_frames):
        img_pil = Image.fromarray(first_frame_rgb.copy())
        draw = ImageDraw.Draw(img_pil)
        
        if i < 30:
            # Phase 1: Slot 2 (E) animates color, Slot 3 (F) is empty
            t = i / 29.0
            
            # Interpolate color
            r = int(c1[0] + (c2[0] - c1[0]) * t)
            g = int(c1[1] + (c2[1] - c1[1]) * t)
            b = int(c1[2] + (c2[2] - c1[2]) * t)
            current_c = (r, g, b)
            
            cx, cy = 518, 682
            w, h = 140, 70
            draw.ellipse([cx - w/2, cy - h/2, cx + w/2, cy + h/2], fill=current_c, outline=(0,0,0), width=2)
            
        else:
            # Phase 2: Slot 2 (E) is static, Slot 3 (F) animates size
            # Draw E (static)
            cx_e, cy_e = 518, 682
            w_e, h_e = 140, 70
            draw.ellipse([cx_e - w_e/2, cy_e - h_e/2, cx_e + w_e/2, cy_e + h_e/2], fill=c2, outline=(0,0,0), width=2)
            
            # Draw F (animating size)
            t = (i - 30) / 29.0
            
            w_f = 140 - (140 - 98) * t
            h_f = 70 - (70 - 49) * t
            
            cx_f, cy_f = 854, 682
            draw.ellipse([cx_f - w_f/2, cy_f - h_f/2, cx_f + w_f/2, cy_f + h_f/2], fill=c2, outline=(0,0,0), width=2)
            
        frames.append(np.array(img_pil))
        
    # Write to output/video.mp4
    # The prompt asks for H.264, yuv420p, 1024x1024, 16 fps.
    # imageio with ffmpeg backend is perfect for this.
    import os
    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    generate_video()
