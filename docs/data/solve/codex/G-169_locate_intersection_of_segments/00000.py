from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    pixels = np.asarray(base)
    fits = []
    for color in [(108, 92, 231), (142, 68, 173)]:
        y, x = np.where(np.all(pixels == color, axis=2))
        fits.append(np.polyfit(x, y, 1))
    (m1,b1),(m2,b2) = fits
    cx = (b2-b1)/(m1-m2)
    cy = m1*cx+b1
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    encoder = subprocess.Popen([
        'ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
        '-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-',
        '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(out/'video.mp4')], stdin=subprocess.PIPE)
    scale = 4
    radius = 33
    for i in range(30):
        frame = base.copy()
        # A brief locating pause, then a continuous clockwise circular stroke.
        if i >= 3:
            progress = min(1.0, (i-2)/25)
            overlay = Image.new('RGBA', (1024*scale,1024*scale))
            draw = ImageDraw.Draw(overlay)
            angles = np.linspace(-np.pi/2, -np.pi/2+2*np.pi*progress, max(2,int(300*progress)))
            pts = [((cx+radius*np.cos(a))*scale,(cy+radius*np.sin(a))*scale) for a in angles]
            red=(230,35,35,255)
            draw.line(pts, fill=red, width=5*scale, joint='curve')
            for x,y in (pts[0],pts[-1]):
                r=2.5*scale
                draw.ellipse((x-r,y-r,x+r,y+r),fill=red)
            overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
            frame.paste(overlay, (0,0), overlay)
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError('ffmpeg encoding failed')
    print(f'Intersection: ({cx:.2f}, {cy:.2f}); wrote {out / "video.mp4"}')

if __name__ == '__main__':
    main()
