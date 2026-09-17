from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import subprocess

ROOT = Path('/app')

def main():
    base = Image.open(ROOT / 'first_frame.png').convert('RGB')
    rgb = np.asarray(base)
    # Recover the filled polygon without changing the reference image.
    mask = ((rgb[:,:,0] < 235) & (rgb[:,:,1] > 225) & (rgb[:,:,2] > 215)).astype(np.uint8)*255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary = max(contours, key=cv2.contourArea)
    vertices = cv2.approxPolyDP(boundary, 2, True).reshape(-1, 2)
    lengths = np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1)
    longest = int(np.argmax(lengths))
    midpoint = (vertices[longest] + vertices[(longest+1)%len(vertices)]) / 2
    # The fill is inset one pixel from the black horizontal outline.
    midpoint[1] -= 1
    print('Edge lengths:', lengths.tolist())
    print('Longest-edge midpoint:', midpoint.tolist())
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    command = ['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
               '-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264',
               '-preset','slow','-crf','0','-pix_fmt','yuv420p',str(out/'video.mp4')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    scale = 4
    x,y = midpoint
    for index in range(25):
        frame = base.copy()
        # Initial pause allows comparison; then draw the midpoint marker steadily.
        if index >= 6:
            fraction = min(1.0, (index-5)/17)
            overlay = Image.new('RGBA',(1024*scale,1024*scale),(0,0,0,0))
            draw = ImageDraw.Draw(overlay)
            r = 10
            box = tuple(int(v*scale) for v in (x-r,y-r,x+r,y+r))
            draw.arc(box, -90, -90+360*fraction, fill=(235,25,35,255), width=3*scale)
            overlay = overlay.resize(base.size,Image.Resampling.LANCZOS)
            frame.paste(overlay, (0,0), overlay)
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
