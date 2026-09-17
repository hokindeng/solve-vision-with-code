import math
import numpy as np
from PIL import Image
import os
import subprocess

def main():
    os.makedirs('/app/frames', exist_ok=True)
    os.makedirs('/app/output', exist_ok=True)

    N = 50
    Thit = 0.25

    img = Image.open('/app/first_frame.png').convert("RGBA")
    bg = np.array(img)
    # Erase Domino 1 and Domino 2 areas
    bg[583:723, 151:197] = [255, 255, 255, 255]
    bg[723:728, 151:197] = [118, 85, 43, 255]
    bg[583:723, 225:271] = [255, 255, 255, 255]
    bg[723:728, 225:271] = [118, 85, 43, 255]
    bg_img = Image.fromarray(bg)

    # Extract D1 and D2
    d1 = np.array(img)[583:725, 151:197]
    d2 = np.array(img)[583:725, 225:271]
    d1_img = Image.fromarray(d1)
    d2_img = Image.fromarray(d2)

    layer1_base = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    layer1_base.paste(d1_img, (151, 583))
    layer2_base = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    layer2_base.paste(d2_img, (225, 583))

    for i in range(N):
        T = i / (N - 1)
        if T <= Thit:
            th2 = 0.0
            th1 = 11.37 * (T / Thit)**2
        else:
            Trel = (T - Thit) / (1.0 - Thit)
            th2 = 90.0 * (Trel)**1.5
            rad2 = math.radians(th2)
            val = (74 * math.cos(rad2) - 46) / 142
            val = max(-1.0, min(1.0, val))
            th1 = th2 + math.degrees(math.asin(val))
        
        rot1 = layer1_base.rotate(-th1, center=(197, 725), resample=Image.BICUBIC)
        rot2 = layer2_base.rotate(-th2, center=(271, 725), resample=Image.BICUBIC)
        
        res = Image.alpha_composite(bg_img, rot2)
        res = Image.alpha_composite(res, rot1)
        
        out_path = f'/app/frames/frame_{i:04d}.png'
        res.convert("RGB").save(out_path)
    
    # Run ffmpeg to compile frames
    cmd = [
        'ffmpeg', '-y', '-framerate', '16', '-i', '/app/frames/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ]
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
