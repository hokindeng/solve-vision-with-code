import cv2
import numpy as np
import subprocess
import os

# Colors
bg_color = np.array([252, 250, 248])
outline_color = np.array([139, 116, 100])

# The shapes and their target offsets (dx, dy)
# Calculated previously using center of mass difference
shapes = [
    {
        'name': 'pink',
        'color': np.array([182, 114, 244]),
        'dx': 566,
        'dy': -52,
        'start_frame': 1,
        'end_frame': 25
    },
    {
        'name': 'red',
        'color': np.array([113, 113, 248]),
        'dx': 423,
        'dy': 162,
        'start_frame': 26,
        'end_frame': 50
    },
    {
        'name': 'yellow',
        'color': np.array([21, 204, 250]),
        'dx': 562,
        'dy': 44,
        'start_frame': 51,
        'end_frame': 75
    }
]

def ease_in_out(t):
    if t <= 0.0: return 0.0
    if t >= 1.0: return 1.0
    return 3 * t**2 - 2 * t**3

def generate_video():
    img = cv2.imread('/app/first_frame.png')
    
    # Extract masks and create a clean background (shapes removed)
    bg = img.copy()
    for s in shapes:
        mask = np.all(img == s['color'], axis=2)
        s['mask'] = mask
        s['y'], s['x'] = np.where(mask)
        bg[mask] = bg_color

    os.makedirs('/app/output/frames', exist_ok=True)
    
    total_frames = 78
    for f in range(total_frames):
        frame_img = bg.copy()
        
        for s in shapes:
            if f < s['start_frame']:
                # Not started moving yet
                cur_dx, cur_dy = 0, 0
            elif f >= s['end_frame']:
                # Finished moving
                cur_dx, cur_dy = s['dx'], s['dy']
            else:
                # Moving
                t = (f - s['start_frame']) / (s['end_frame'] - s['start_frame'])
                progress = ease_in_out(t)
                cur_dx = int(round(s['dx'] * progress))
                cur_dy = int(round(s['dy'] * progress))
                
            # Draw shape
            ty, tx = s['y'] + cur_dy, s['x'] + cur_dx
            frame_img[ty, tx] = s['color']
            
        cv2.imwrite(f'/app/output/frames/frame_{f:04d}.png', frame_img)

    # Encode to mp4 using ffmpeg
    subprocess.run([
        'ffmpeg', '-y', '-framerate', '16', '-i', '/app/output/frames/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ], check=True)

if __name__ == '__main__':
    generate_video()
