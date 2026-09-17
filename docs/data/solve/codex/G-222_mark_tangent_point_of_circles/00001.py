from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/app')

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    pixels = np.array(base)
    circles = []
    for color in [(52,152,219), (142,68,173)]:
        yy, xx = np.where(np.all(pixels == color, axis=2))
        center = np.array([(xx.min()+xx.max())/2, (yy.min()+yy.max())/2])
        radius = ((xx.max()-xx.min())+(yy.max()-yy.min()))/4
        circles.append((center,radius))
    (a,ra),(b,rb) = circles
    tangent = a + (b-a)*ra/(ra+rb)
    out = ROOT/'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','10','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')],stdin=subprocess.PIPE)
    scale = 4
    x,y = tangent
    r = 22
    for i in range(60):
        frame = base.copy()
        if i > 3:
            progress = min(1.0,(i-3)/52)
            overlay = Image.new('RGBA',(1024*scale,1024*scale))
            draw = ImageDraw.Draw(overlay)
            box = tuple(v*scale for v in (x-r,y-r,x+r,y+r))
            draw.arc(box,start=-90,end=-90+360*progress,fill=(0,0,0,255),width=4*scale)
            overlay = overlay.resize(base.size,Image.Resampling.LANCZOS)
            frame.paste(overlay,(0,0),overlay)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
