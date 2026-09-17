from pathlib import Path
import math
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/app')
OUT = ROOT / 'output/video.mp4'
FPS, FRAMES = 16, 192
G, ELASTICITY, H0, V0 = 10.0, 0.70, 11.2, 3.6

def trajectory():
    segments = []
    t, h, v = 0.0, H0, V0
    while True:
        duration = (v + math.sqrt(v*v + 2*G*h))/G
        segments.append((t, t + duration, h, v))
        impact = v - G*duration
        t += duration
        v = -ELASTICITY * impact
        h = 0.0
        if v < 0.18:
            return segments, t


def main():
    OUT.parent.mkdir(exist_ok=True)
    original = Image.open(ROOT/'first_frame.png').convert('RGB')
    background = original.copy()
    # Remove only the initial moving objects; their backdrop is plain white.
    ImageDraw.Draw(background).rectangle((365, 320, 595, 418), fill='white')
    ball = original.crop((531, 355, 592, 416))
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 30)
    segments, stop_time = trajectory()
    # Spread the diminishing bounces across nearly the entire clip.
    motion_duration = 10.8
    process = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1024x1024',
        '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264',
        '-preset', 'slow', '-crf', '17', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(OUT)
    ], stdin=subprocess.PIPE)
    for frame in range(FRAMES):
        if frame == 0:
            image = original
        else:
            t = frame/FPS * stop_time/motion_duration
            h, v = 0.0, 0.0
            for start, end, height, velocity in segments:
                if start <= t < end:
                    dt = t-start
                    h = max(0.0, height + velocity*dt - 0.5*G*dt*dt)
                    v = velocity-G*dt
                    break
            image = background.copy()
            cy = round(744 - (359/H0)*h)
            image.paste(ball, (531, cy-30))
            draw = ImageDraw.Draw(image)
            green = (60, 180, 60)
            length = round(abs(v)*6.4)
            if abs(v) >= 0.05:
                length = max(7, length)
                if v > 0:
                    base, tip = cy-35, cy-35-length
                    draw.line((561, base, 561, tip+7), fill=green, width=5)
                    draw.polygon([(561, tip), (555, tip+13), (567, tip+13)], fill=green)
                    label_y = cy-67
                else:
                    base, tip = cy+35, cy+35+length
                    draw.line((561, base, 561, tip-7), fill=green, width=5)
                    draw.polygon([(561, tip), (555, tip-13), (567, tip-13)], fill=green)
                    label_y = cy+33
            else:
                label_y = cy-67
            text = f'v={abs(v):.1f} m/s'
            width = draw.textlength(text, font=font)
            draw.text((536-width, label_y), text, font=font, fill=green)
        process.stdin.write(np.asarray(image).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
