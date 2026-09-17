from PIL import Image, ImageDraw
import numpy as np
import subprocess
import math

SOURCE = '/app/first_frame.png'
OUTPUT = '/app/output/video.mp4'

def main():
    base = Image.open(SOURCE).convert('RGB')
    w, h = base.size
    proc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', '16', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', OUTPUT], stdin=subprocess.PIPE)
    for i in range(48):
        frame = base.copy()
        if i:
            progress = min(i / 44, 1.0)
            scale = 4
            overlay = Image.new('RGBA', (w*scale, h*scale), (0,0,0,0))
            draw = ImageDraw.Draw(overlay)
            n = max(2, int(500 * progress))
            angles = np.linspace(-math.pi/2, -math.pi/2 + 2*math.pi*progress, n)
            points = [((171 + 94*math.cos(a))*scale, (512 + 94*math.sin(a))*scale) for a in angles]
            draw.line(points, fill=(230, 20, 25, 255), width=5*scale, joint='curve')
            for x,y in (points[0], points[-1]):
                draw.ellipse((x-2.5*scale,y-2.5*scale,x+2.5*scale,y+2.5*scale),fill=(230,20,25,255))
            overlay = overlay.resize((w,h),Image.Resampling.LANCZOS)
            frame.paste(overlay, (0,0), overlay)
        proc.stdin.write(np.asarray(frame).tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
