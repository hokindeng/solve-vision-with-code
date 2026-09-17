from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    background = original.copy()
    # Uncover the right-hand position markers beneath the two moving handles.
    for start, shift in [(253, -307), (867, 307)]:
        background[648:705, start:start+57] = 0
        background[672:681, 589+shift:598+shift] = original[672:681,589:598]
    yy, xx = np.indices(original.shape[:2])
    purple = np.all(original == (128,0,128), axis=2)
    masks = [purple & (xx < 350), purple & (xx > 700)]
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    command = ['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
               '-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264',
               '-preset','slow','-crf','0','-pix_fmt','yuv420p','-movflags','+faststart',
               str(output / 'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in range(24):
        if frame == 0:
            canvas = original.copy()
        else:
            canvas = background.copy()
            for index, (start, end, first, last) in enumerate([(253,176,2,11),(867,790,12,21)]):
                t = float(np.clip((frame-first)/(last-first),0,1))
                eased = t*t*(3-2*t)
                x = round(start + (end-start)*eased)
                canvas[648:705,x:x+57] = (128,128,128)
                if t == 1:
                    canvas[masks[index]] = (0,0,255)
        process.stdin.write(canvas.tobytes())
    process.stdin.close()
    error = process.stderr.read()
    if process.wait():
        raise RuntimeError(error.decode())

if __name__ == '__main__':
    main()
