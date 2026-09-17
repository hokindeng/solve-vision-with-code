from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image

ROOT = Path('/app')

def main():
    original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    black_mask = np.all(original == (0, 0, 0), axis=2)
    background = original.copy()
    background[black_mask] = 255
    # Preserve every stationary colored ball using its original pixels.
    targets = []
    for color, requested_center in [((255,215,0),(268,847)), ((60,179,113),(769,422)), ((255,215,0),(106,306)), ((220,20,60),(910,626))]:
        mask = np.all(original == color, axis=2).astype(np.uint8)
        _, labels, _, centers = cv2.connectedComponentsWithStats(mask)
        component = min(range(1,len(centers)), key=lambda k: np.linalg.norm(centers[k]-requested_center))
        targets.append((np.array(requested_center, dtype=float), labels == component))
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    encoder = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','15','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')], stdin=subprocess.PIPE)
    radii = [38.5,64.0,85.0,110.0,145.0]
    start = np.array([343.,328.])
    for frame in range(108):
        if frame < 4:
            canvas = original.copy()
        else:
            stage = min((frame-4)//24,3)
            local = min((frame-4)-24*stage,23)
            dest = targets[stage][0]
            origin = start if stage == 0 else targets[stage-1][0]
            travel = min(local/17.0,1.0)
            ease = travel*travel*(3-2*travel)
            center = origin + (dest-origin)*ease
            grow = max(0., min((local-17)/6.,1.))
            grow = grow*grow*(3-2*grow)
            radius = radii[stage] + (radii[stage+1]-radii[stage])*grow
            if stage == 3:
                center = center + np.array([-50.,0.])*grow
            canvas = background.copy()
            for j, (_, mask) in enumerate(targets):
                if j < stage or (j == stage and local >= 17):
                    canvas[mask] = 255
            # Draw only the moving black ball; original background and remaining
            # colored-ball pixels are retained verbatim outside its silhouette.
            x0=max(0,int(center[0]-radius-2)); x1=min(1024,int(center[0]+radius+3))
            y0=max(0,int(center[1]-radius-2)); y1=min(1024,int(center[1]+radius+3))
            yy,xx=np.mgrid[y0:y1,x0:x1]
            alpha=np.clip(radius+0.5-np.sqrt((xx-center[0])**2+(yy-center[1])**2),0,1)
            canvas[y0:y1,x0:x1]=(canvas[y0:y1,x0:x1]*(1-alpha[:,:,None])).round().astype(np.uint8)
        encoder.stdin.write(canvas.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('Video encoding failed')

if __name__ == '__main__':
    main()
