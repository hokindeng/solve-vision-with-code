from pathlib import Path
import cv2
import numpy as np
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output' / 'video.mp4'

def main():
    source = cv2.imread(str(ROOT / 'first_frame.png'))
    pipe_color = np.array([233, 165, 14], dtype=np.uint8)
    pipes = np.all(source == pipe_color, axis=2)
    background = source.copy()
    background[pipes] = 255
    # Clockwise screen-space angles. Each elbow turns about its tile center.
    # The final ports point inward around the four-tile circuit.
    tiles = [(387,387,360), (637,387,-100),
             (387,637,-170), (637,637,175)]
    layers = []
    for x,y,turn in tiles:
        mask = np.zeros((1024,1024), np.uint8)
        mask[y-107:y+108,x-107:x+108] = pipes[y-107:y+108,x-107:x+108] * 255
        layers.append((x,y,turn,mask))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(['ffmpeg','-y','-loglevel','error',
        '-f','rawvideo','-pix_fmt','bgr24','-s','1024x1024','-r','16',
        '-i','-','-an','-c:v','libx264','-preset','slow','-crf','16',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(OUT)], stdin=subprocess.PIPE)
    for frame_index in range(96):
        t = frame_index / 95
        progress = t*t*(3-2*t)
        frame = background.copy()
        for x,y,turn,mask in layers:
            transform = cv2.getRotationMatrix2D((x,y), -turn*progress, 1)
            rotated = cv2.warpAffine(mask, transform, (1024,1024), flags=cv2.INTER_NEAREST)
            frame[rotated > 0] = pipe_color
        if frame_index == 0:
            frame = source.copy()
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
