from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')

def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t*t*(3-2*t)

def main():
    base = np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
    # Copy the reference star's exact raster, retaining its shape and color.
    region = base[20:137, 888:1006]
    mask = (region[:,:,2] > 200) & (region[:,:,0] < 80) & (region[:,:,1] < 80)
    sy, sx = np.where(mask)
    colors = region[sy, sx].astype(float)
    sx = sx + 888 - 947 + 722
    sy = sy + 20 - 78 + 512
    out = ROOT/'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo',
        '-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16',
        '-i','-','-an','-c:v','libx264','-preset','slow','-crf','0',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')], stdin=subprocess.PIPE)
    for i in range(24):
        frame = base.copy()
        t = i/23
        # The last slot is empty, so no existing symbols need displacement.
        opacity = smooth(t/0.38)
        drop = smooth((t-0.32)/0.64)
        yy = sy - round(125*(1-drop))
        frame[yy,sx] = np.rint(frame[yy,sx]*(1-opacity) + colors*opacity).astype(np.uint8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
