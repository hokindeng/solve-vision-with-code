from pathlib import Path
import math
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output' / 'video.mp4'
FPS, FRAMES, SCALE = 16, 96, 4

def main():
    OUT.parent.mkdir(exist_ok=True)
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    background = original.copy()
    orange = (original[:,:,0] == 249) & (original[:,:,1] == 115) & (original[:,:,2] == 22)
    background[orange] = (255,255,255)
    # Centers, initial angles, and simultaneous clockwise rotations.
    # The two already aligned elbows make a full turn, so every pipe moves.
    pipes = [(387.5,387.5,0,360), (637.5,387.5,70,380),
             (387.5,637.5,270,360), (637.5,637.5,120,420)]
    # A square-ended elbow, with its junction at the tile center.
    polygon = np.array([[-7.5,-7.5],[103.5,-7.5],[103.5,7.5],
                        [7.5,7.5],[7.5,103.5],[-7.5,103.5]])
    command = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
               '-pix_fmt','rgb24','-s','1024x1024','-r',str(FPS),'-i','-',
               '-an','-c:v','libx264','-crf','17','-preset','slow','-pix_fmt','yuv420p',
               '-movflags','+faststart',str(OUT)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for i in range(FRAMES):
        if i == 0:
            frame = original
        else:
            t = i / (FRAMES-1)
            progress = t*t*(3-2*t)
            frame = background.copy()
            for cx,cy,start,turn in pipes:
                angle = math.radians(start + turn*progress)
                matrix = np.array([[math.cos(angle),-math.sin(angle)],
                                   [math.sin(angle),math.cos(angle)]])
                points = polygon @ matrix.T + [110.5,110.5]
                mask = Image.new('L',(221*SCALE,221*SCALE),0)
                ImageDraw.Draw(mask).polygon([tuple(p*SCALE) for p in points],fill=255)
                alpha = np.asarray(mask.resize((221,221),Image.Resampling.LANCZOS),dtype=float)/255
                x,y = int(cx-110.5),int(cy-110.5)
                patch = frame[y:y+221,x:x+221]
                patch[:] = np.rint(patch*(1-alpha[:,:,None]) + np.array([249,115,22])*alpha[:,:,None]).astype(np.uint8)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError('ffmpeg encoding failed')

if __name__ == '__main__':
    main()
