import cv2
import numpy as np
import subprocess
import os

def smoothstep(x):
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    return x * x * (3.0 - 2.0 * x)

def make_video():
    first_frame = cv2.imread('/app/first_frame.png')
    h, w, _ = first_frame.shape

    # Extract patches
    patch_E3 = first_frame[465:465+95, 253+2:253+97].copy()
    patch_E4 = first_frame[465:465+95, 358+2:358+97].copy()
    patch_E5 = first_frame[465:465+95, 463+2:463+97].copy()
    patch_T = patch_E4.copy() # Triangle to insert

    os.makedirs('/app/output', exist_ok=True)
    writer = cv2.VideoWriter('/app/output/video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 16, (w, h))

    for frame_idx in range(54):
        img = first_frame.copy()

        # Erase original E3, E4, E5 inner boxes
        img[465:465+95, 253+2:253+97] = [255, 255, 255]
        img[465:465+95, 358+2:358+97] = [255, 255, 255]
        img[465:465+95, 463+2:463+97] = [255, 255, 255]

        # Calculate time parameters
        slide_t = 0.0
        if frame_idx >= 6 and frame_idx <= 30:
            slide_t = (frame_idx - 6) / 24.0
        elif frame_idx > 30:
            slide_t = 1.0
        slide_t = smoothstep(slide_t)

        fade_t = 0.0
        if frame_idx >= 31 and frame_idx <= 45:
            fade_t = (frame_idx - 31) / 14.0
        elif frame_idx > 45:
            fade_t = 1.0
        
        # Calculate positions
        x3 = 253 + (358 - 253) * slide_t
        x4 = 358 + (568 - 358) * slide_t
        x5 = 463 + (673 - 463) * slide_t

        # Draw sliding symbols
        def draw_sym(patch, box_x_float):
            draw_x = int(round(box_x_float)) + 2
            mask = np.any(patch != [255, 255, 255], axis=-1)
            img[465:465+95, draw_x:draw_x+95][mask] = patch[mask]

        draw_sym(patch_E3, x3)
        draw_sym(patch_E4, x4)
        draw_sym(patch_E5, x5)

        # Draw fading inserted symbols
        if fade_t > 0:
            def fade_in(patch, box_x):
                draw_x = box_x + 2
                mask = np.any(patch != [255, 255, 255], axis=-1)
                patch_faded = (patch.astype(float) * fade_t + 255.0 * (1 - fade_t)).astype(np.uint8)
                img[465:465+95, draw_x:draw_x+95][mask] = patch_faded[mask]

            fade_in(patch_T, 253) # pos 3
            fade_in(patch_T, 463) # pos 5

        writer.write(img)

    writer.release()
    
    # re-encode with ffmpeg for correct pixel format and codec
    subprocess.run([
        'ffmpeg', '-y', '-i', '/app/output/video.mp4',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/final_video.mp4'
    ], check=True)
    os.replace('/app/output/final_video.mp4', '/app/output/video.mp4')

if __name__ == '__main__':
    make_video()
