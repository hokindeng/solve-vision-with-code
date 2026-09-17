from pathlib import Path
import math
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent

def main():
    base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    h, w = base.shape[:2]
    yy, xx = np.indices((h, w))
    gray = (base[:,:,0] == base[:,:,1]) & (base[:,:,1] == base[:,:,2])
    arc = (xx >= 472) & (xx <= 513) & (yy >= 470) & (yy <= 500) & ((xx < 510) | (yy < 480))
    label = (xx >= 558) & (xx <= 648) & (yy >= 467) & (yy <= 490)
    erase = gray & (arc | label) & (base[:,:,0] < 255)
    angle = math.asin(math.sin(math.radians(69.8)) / 1.850)
    start = np.array([512., 512.])
    end = np.array([512. + (h-1-512.)*math.tan(angle), h-1.])
    direction = np.array([math.sin(angle), math.cos(angle)])
    perpendicular = np.array([-direction[1], direction[0]])
    (ROOT / 'output').mkdir(exist_ok=True)
    proc = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{w}x{h}','-framerate','16','-i','-','-an','-c:v','libx264','-crf','0','-preset','slow','-pix_fmt','yuv420p','-movflags','+faststart',str(ROOT/'output/video.mp4')], stdin=subprocess.PIPE)
    for i in range(70):
        frame = base.copy()
        fade = min(i/12., 1.)
        frame[erase] = np.round(base[erase].astype(float)*(1-fade)+255*fade).astype(np.uint8)
        progress = max(0., (i-12)/57.)
        if progress > 0:
            tip = start + progress*(end-start)
            scale = 4
            mask = Image.new('L', (w*scale, h*scale))
            draw = ImageDraw.Draw(mask)
            def coords(p): return tuple(float(v*scale) for v in p)
            draw.line([coords(start), coords(tip)], fill=255, width=6)
            arrow = min(14., float(np.linalg.norm(tip-start))*.35)
            for sign in [-1,1]:
                wing = tip - direction*arrow + sign*perpendicular*arrow*.42
                draw.line([coords(wing),coords(tip)],fill=255,width=6)
            alpha = np.asarray(mask.resize((w,h),Image.Resampling.LANCZOS)).astype(float)/255
            alpha[yy < 512] = 0
            alpha = alpha[:,:,None]
            frame = np.round(frame*(1-alpha) + np.array([255,0,0])*alpha).astype(np.uint8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0: raise RuntimeError('ffmpeg failed')
    print(f'Refraction angle: {math.degrees(angle):.4f} degrees; endpoint: {end}')

if __name__ == '__main__':
    main()
