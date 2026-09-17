import cv2
import numpy as np
import os
import subprocess
import shutil

def main():
    os.makedirs('/app/output', exist_ok=True)
    os.makedirs('/app/frames', exist_ok=True)

    img = cv2.imread('/app/first_frame.png')

    num_frames = 96
    fps = 16

    tiles_info = [
        {'x': 276, 'y': 276, 'target_angle': 0},
        {'x': 526, 'y': 276, 'target_angle': 90},
        {'x': 276, 'y': 526, 'target_angle': 180},
        {'x': 526, 'y': 526, 'target_angle': 180},
    ]

    Y, X = np.ogrid[:222, :222]
    dist = np.sqrt((X - 111)**2 + (Y - 111)**2)
    mask = dist <= 106
    mask_3d = mask[:, :, None]

    for f in range(num_frames):
        frame = img.copy()
        
        # Linear progress
        p = f / (num_frames - 1)
        # Ease-in-out progress (smoothstep)
        progress = 3 * (p ** 2) - 2 * (p ** 3)
        
        for info in tiles_info:
            x, y = info['x'], info['y']
            target_angle = info['target_angle']
            
            current_angle = target_angle * progress
            
            if current_angle != 0:
                tile = frame[y:y+222, x:x+222]
                M = cv2.getRotationMatrix2D((111, 111), current_angle, 1.0)
                rotated = cv2.warpAffine(tile, M, (222, 222), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
                
                final_tile = np.where(mask_3d, rotated, tile)
                frame[y:y+222, x:x+222] = final_tile
                
        cv2.imwrite(f'/app/frames/frame_{f:04d}.png', frame)

    subprocess.run([
        'ffmpeg', '-y', '-framerate', str(fps), '-i', '/app/frames/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '/app/output/video.mp4'
    ], check=True)

    # Clean up frames
    shutil.rmtree('/app/frames')

if __name__ == '__main__':
    main()
