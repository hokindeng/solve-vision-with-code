from PIL import Image, ImageDraw
import numpy as np
import subprocess
from pathlib import Path

ROOT = Path('/app')
SOURCE = ROOT / 'first_frame.png'
OUTPUT = ROOT / 'output/video.mp4'

def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    original = Image.open(SOURCE).convert('RGB')
    # The example changes only outline weight: retain the unfilled octagon.
    vertices = [(864,688),(922,711),(945,768),(922,826),
                (864,849),(807,826),(784,768),(807,711),(864,688)]
    lengths = [np.hypot(b[0]-a[0],b[1]-a[1]) for a,b in zip(vertices,vertices[1:])]
    question_box = (838,738,890,798)
    question = original.crop(question_box)
    white = Image.new('RGB', question.size, 'white')
    proc = subprocess.Popen(['ffmpeg','-y','-loglevel','error',
        '-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
        '-s','1024x1024','-r','16','-i','-','-an',
        '-c:v','libx264','-preset','slow','-crf','0','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(OUTPUT)], stdin=subprocess.PIPE)
    for i in range(60):
        frame = original.copy()
        if i:
            fade = min(1.0, i / 12)
            frame.paste(Image.blend(question, white, fade), question_box)
            progress = max(0.0, (i-10)/49)
            distance = sum(lengths)*progress
            draw = ImageDraw.Draw(frame)
            for a,b,length in zip(vertices,vertices[1:],lengths):
                if distance <= 0:
                    break
                fraction = min(1.0,distance/length)
                endpoint = (round(a[0]+(b[0]-a[0])*fraction),round(a[1]+(b[1]-a[1])*fraction))
                draw.line([a,endpoint], fill=(70,153,53), width=1)
                distance -= length
        proc.stdin.write(np.asarray(frame).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
