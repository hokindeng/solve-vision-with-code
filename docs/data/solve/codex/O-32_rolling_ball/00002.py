from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    teal = ((original[:,:,0] < 100) & (original[:,:,1] > 80) & (original[:,:,2] > 90)).astype(np.uint8)
    n, labels, stats, centers = cv2.connectedComponentsWithStats(teal)
    platforms = sorted([tuple(centers[i]) for i in range(1,n) if stats[i,4] > 100])
    # Isolate the original ball, retaining its exact fill, outline, and edge pixels.
    ball_mask = np.zeros(original.shape[:2], np.uint8)
    region = original[540:630,320:412]
    ball_mask[540:630,320:412] = np.any(region < 250, axis=2).astype(np.uint8)
    ys, xs = np.where(ball_mask)
    start = np.array([(xs.min()+xs.max())/2, (ys.min()+ys.max())/2])
    background = original.copy()
    background[ball_mask.astype(bool)] = 255
    alpha = ball_mask.astype(np.float32)
    premult = original.astype(np.float32) * alpha[:,:,None]
    # The ball's lower rim meets each platform's upper surface.
    positions = [start] + [np.array([x,y-50.0]) for x,y in platforms]
    times = [0,10] + list(range(15,61,5))
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    command = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
               '-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264',
               '-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(output/'video.mp4')]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame in range(64):
        if frame == 0:
            result = original
        else:
            if frame >= times[-1]:
                pos = positions[-1]
            else:
                segment = np.searchsorted(times, frame, side='right') - 1
                u = (frame-times[segment])/(times[segment+1]-times[segment])
                smooth = u*u*(3-2*u)
                pos = positions[segment]*(1-smooth) + positions[segment+1]*smooth
                pos = pos.copy()
                pos[1] -= (13 if segment == 0 else 5) * np.sin(np.pi*u)**2
            dx,dy = pos-start
            transform = np.float32([[1,0,dx],[0,1,dy]])
            a = cv2.warpAffine(alpha,transform,(1024,1024),flags=cv2.INTER_LINEAR)
            color = cv2.warpAffine(premult,transform,(1024,1024),flags=cv2.INTER_LINEAR)
            result = np.clip(color+background.astype(np.float32)*(1-a[:,:,None]),0,255).astype(np.uint8)
        proc.stdin.write(result.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
