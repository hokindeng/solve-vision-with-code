from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')

def main():
    original = Image.open(ROOT / 'first_frame.png').convert('RGB')
    pixels = np.asarray(original)
    orange = (pixels[:,:,0] > 180) & (pixels[:,:,1] < 130) & (pixels[:,:,2] < 80)
    purple = (pixels[:,:,0] > 60) & (pixels[:,:,0] < 160) & (pixels[:,:,1] < 130) & (pixels[:,:,2] > 170)
    fits = []
    for mask in (orange, purple):
        y, x = np.where(mask)
        fits.append(np.polyfit(y, x, 1))
    (a,b),(c,d) = fits
    cy = (d-b)/(a-c)
    cx = a*cy+b
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    encoder = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','15','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')], stdin=subprocess.PIPE)
    scale = 4
    radius = 32
    for i in range(30):
        frame = original.copy()
        if i >= 3:
            progress = min(1.0, (i-2)/25)
            layer = Image.new('RGBA', (1024*scale,1024*scale))
            draw = ImageDraw.Draw(layer)
            angles = np.linspace(-np.pi/2, -np.pi/2 + 2*np.pi*progress, max(2,int(240*progress)))
            points = [((cx+radius*np.cos(t))*scale, (cy+radius*np.sin(t))*scale) for t in angles]
            draw.line(points, fill=(235,25,35,255), width=5*scale, joint='curve')
            for x,y in (points[0],points[-1]):
                r=2.5*scale
                draw.ellipse((x-r,y-r,x+r,y+r), fill=(235,25,35,255))
            layer = layer.resize(original.size, Image.Resampling.LANCZOS)
            frame.paste(layer, (0,0), layer)
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError('ffmpeg failed')
    print(f'Intersection: ({cx:.2f}, {cy:.2f})')

if __name__ == '__main__':
    main()
