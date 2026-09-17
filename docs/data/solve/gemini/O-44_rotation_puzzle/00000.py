import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    tiles_info = [
        {"rect": (276, 276), "diff": 0},
        {"rect": (526, 276), "diff": -66},
        {"rect": (276, 526), "diff": -36},
        {"rect": (526, 526), "diff": -163}
    ]
    
    mask = np.zeros((222, 222), dtype=np.uint8)
    cv2.circle(mask, (111, 111), 106, 255, -1)
    
    os.makedirs('/app/output', exist_ok=True)
    frames = []
    num_frames = 96
    
    for i in range(num_frames):
        progress = i / (num_frames - 1)
        # Add smoothstep for better pacing
        progress = progress * progress * (3 - 2 * progress)
        
        frame = img_rgb.copy()
        for info in tiles_info:
            x, y = info["rect"]
            tile = img_rgb[y:y+222, x:x+222]
            
            angle = progress * info["diff"]
            if angle == 0:
                continue
                
            M = cv2.getRotationMatrix2D((111, 111), angle, 1.0)
            # Use INTER_CUBIC for better quality during rotation
            rotated = cv2.warpAffine(tile, M, (222, 222), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
            
            tile_final = np.where(mask[:, :, None] == 255, rotated, tile)
            frame[y:y+222, x:x+222] = tile_final
            
        frames.append(frame)
        
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', macro_block_size=None, pixelformat='yuv420p')
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == "__main__":
    solve()
