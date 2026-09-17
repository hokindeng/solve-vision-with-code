from pathlib import Path
import math
import cv2
import numpy as np
from PIL import Image
import imageio.v2 as imageio

ROOT = Path('/app')
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
source = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
center = np.array([572., 589.])
vertices = np.array([[572.,589.], [560.,545.], [635.,478.], [658.,565.]])

def rotate(angle):
    a = math.radians(angle)
    # Positive angles are counterclockwise in image coordinates.
    matrix = np.array([[math.cos(a),math.sin(a)],[-math.sin(a),math.cos(a)]])
    return (vertices-center) @ matrix.T + center

def dashed_polygon(canvas, points):
    for start, stop in zip(points, np.roll(points,-1,axis=0)):
        length = np.linalg.norm(stop-start)
        direction = (stop-start)/length
        for d in np.arange(0,length,16):
            p = np.rint(start+direction*d).astype(int)
            q = np.rint(start+direction*min(d+10,length)).astype(int)
            cv2.line(canvas,tuple(p),tuple(q),(100,100,100),3,cv2.LINE_8)

# Remove only the original polygon; retain every visible target-outline pixel.
polygon_mask = np.all(source == [77,147,189],axis=2) | np.all(source == [50,50,50],axis=2)
base = source.copy()
base[polygon_mask] = [240,240,240]
# Restore the part of the target originally hidden by the filled polygon.
target = np.full_like(source,240)
dashed_polygon(target,rotate(42))
base[polygon_mask] = target[polygon_mask]
marker_mask = np.all(source == [0,0,0],axis=2) | np.all(source == [255,255,255],axis=2)

with imageio.get_writer(OUT/'video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=1, ffmpeg_params=['-crf','18']) as writer:
    for i in range(70):
        if i <= 3:
            frame = source.copy()
        else:
            t = np.clip((i-3)/60,0,1)
            progress = t*t*(3-2*t)
            frame = base.copy()
            points = np.rint(rotate(42*progress)).astype(np.int32)
            cv2.fillPoly(frame,[points],(77,147,189),lineType=cv2.LINE_8)
            cv2.polylines(frame,[points],True,(50,50,50),1,cv2.LINE_8)
            frame[marker_mask] = source[marker_mask]
        writer.append_data(frame)
