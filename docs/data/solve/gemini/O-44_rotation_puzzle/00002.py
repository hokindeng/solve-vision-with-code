import cv2
import numpy as np
import imageio

def generate_solution():
    # Read the first frame
    img = cv2.imread('/app/first_frame.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    frames = []
    num_frames = 96
    fps = 16

    # The tiles coordinates and their required rotation to solve the puzzle.
    # The target loop is:
    # TL: Bottom, Right
    # TR: Bottom, Left
    # BL: Top, Right
    # BR: Top, Left
    
    tiles_info = {
        "TL": (276, 276, 180),  # Currently Top, Left. Needs 180 to become Bottom, Right.
        "TR": (526, 276, -90), # Currently Bottom, Right. Needs -90 (CW) to become Bottom, Left.
        "BL": (276, 526, 0),   # Currently Top, Right. Needs 0 rotation.
        "BR": (526, 526, 0)    # Currently Top, Left. Needs 0 rotation.
    }

    # Mask for the border of the tiles. We only want to rotate the inner pipe area.
    # The border is at indices 0-3 and 219-221.
    border_mask = np.ones((222, 222, 3), dtype=bool)
    border_mask[4:219, 4:219, :] = False

    for i in range(num_frames):
        # Linear progress from 0.0 to 1.0
        progress = i / (num_frames - 1)
        
        frame = img_rgb.copy()
        
        for name, (x, y, total_angle) in tiles_info.items():
            angle = total_angle * progress
            
            # Extract the tile from the original image
            tile_orig = img_rgb[y:y+222, x:x+222]
            
            # Rotate the entire tile
            M = cv2.getRotationMatrix2D((111, 111), angle, 1.0)
            rotated_tile = cv2.warpAffine(tile_orig, M, (222, 222), borderValue=(255, 255, 255), flags=cv2.INTER_LINEAR)
            
            # Restore the unrotated border so only the inside appears to rotate
            np.copyto(rotated_tile, tile_orig, where=border_mask)
            
            # Put the tile back into the frame
            frame[y:y+222, x:x+222] = rotated_tile
            
        frames.append(frame)

    # Write the video to /app/output/video.mp4
    # The task asks for H.264, yuv420p, 1024x1024, 16 fps
    imageio.mimwrite('/app/output/video.mp4', frames, fps=fps, format='FFMPEG', codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
if __name__ == '__main__':
    generate_solution()
