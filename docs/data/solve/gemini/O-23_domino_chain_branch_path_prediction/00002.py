import cv2
import numpy as np
from PIL import Image
import math
import subprocess
import os

def render_video():
    # Load original image
    img_path = '/app/first_frame.png'
    orig = Image.open(img_path).convert('RGB')
    
    # Define dominos
    # Format: id: (x, y)
    domino_coords = {
        0: (74, 440),
        1: (196, 440),
        2: (318, 440),
        3: (440, 440),
        4: (562, 332),
        5: (684, 314),
        6: (806, 296),
        7: (928, 279),
        8: (562, 548),
        9: (684, 565)
    }
    
    w, h = 42, 103
    bg_color = (240, 235, 230)
    
    # Create background by erasing dominos
    bg = orig.copy()
    from PIL import ImageDraw
    draw = ImageDraw.Draw(bg)
    for idx, (dx, dy) in domino_coords.items():
        draw.rectangle([dx, dy, dx+w-1, dy+h-1], fill=bg_color)
        
    # Extract domino patches with transparency
    # We will just extract them as RGBA
    domino_patches = {}
    orig_rgba = orig.convert('RGBA')
    for idx, (dx, dy) in domino_coords.items():
        patch = orig_rgba.crop((dx, dy, dx+w, dy+h))
        domino_patches[idx] = patch

    # Animation parameters
    fps = 16
    total_frames = 62
    t_start = 3
    delay = 6
    duration = 15
    max_angle = 70.0
    
    stage_map = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        8: 4,
        5: 5,
        9: 5,
        6: 6,
        7: 7
    }
    
    os.makedirs('/app/frames', exist_ok=True)
    
    for frame_idx in range(total_frames):
        # Create a new frame based on bg
        frame = bg.copy().convert('RGBA')
        
        # Draw dominos in reverse x order so left dominos rest on right dominos
        sorted_dominos = sorted(domino_coords.items(), key=lambda item: item[1][0], reverse=True)
        for idx, (dx, dy) in sorted_dominos:
            stage = stage_map[idx]
            start_frame = t_start + stage * delay
            
            if frame_idx <= start_frame:
                progress = 0.0
            else:
                progress = (frame_idx - start_frame) / duration
                if progress > 1.0:
                    progress = 1.0
                    
            # Ease-out sine
            angle = max_angle * math.sin(progress * math.pi / 2.0)
            
            patch = domino_patches[idx]
            
            if angle == 0:
                frame.alpha_composite(patch, (dx, dy))
            else:
                # Rotate the patch around its bottom-right corner
                # In PIL, rotate rotates around the center by default, and expands the image if expand=True
                # It's easier to rotate with expand=True, then calculate where to paste it.
                rotated = patch.rotate(-angle, resample=Image.BICUBIC, expand=True) # negative for clockwise
                
                # Pivot in original patch
                px = w
                py = h
                
                # To find the new position, we can just use the pivot's position in the rotated image.
                # When PIL rotates with expand=True, the image is rotated around its center.
                # Let's do it manually with affine transform, or calculate the shift.
                
                # The original center is (w/2, h/2).
                cx, cy = w / 2, h / 2
                
                # The pivot relative to center:
                rel_px = px - cx
                rel_py = py - cy
                
                # Rotate this relative pivot by -angle
                rad = math.radians(angle)
                cos_a = math.cos(rad)
                sin_a = math.sin(rad)
                
                # clockwise rotation matrix
                rot_rel_px = rel_px * cos_a - rel_py * sin_a
                rot_rel_py = rel_px * sin_a + rel_py * cos_a
                
                # The new center in the rotated image is (rw/2, rh/2)
                rw, rh = rotated.size
                rcx, rcy = rw / 2.0, rh / 2.0
                
                # So the pivot in the rotated image is at:
                pivot_rot_x = rcx + rot_rel_px
                pivot_rot_y = rcy + rot_rel_py
                
                # We want this pivot_rot_x, pivot_rot_y to be placed at the original pivot position on screen: (dx + px, dy + py)
                # So the top-left of the rotated image should be placed at:
                paste_x = (dx + px) - pivot_rot_x
                paste_y = (dy + py) - pivot_rot_y
                
                # Round to integer for pasting
                paste_x = int(round(paste_x))
                paste_y = int(round(paste_y))
                
                frame.alpha_composite(rotated, (paste_x, paste_y))
                
        # Save frame
        out_path = f'/app/frames/frame_{frame_idx:04d}.png'
        frame.convert('RGB').save(out_path)
        
    # Compile video
    cmd = [
        'ffmpeg', '-y', '-framerate', str(fps),
        '-i', '/app/frames/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '/app/output/video.mp4'
    ]
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    render_video()
