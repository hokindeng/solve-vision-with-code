from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
original = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
# Rows cycle star, notched diamond, plus; the missing item is the diamond.
# Copy the existing diamond's pixels so its stroke matches exactly.
target = original.copy()
target[795:912, 790:916] = 255
target[817:918, 791:921] = original[135:236, 450:580]
question = np.zeros((1024, 1024), dtype=bool)
question[795:912,790:916] = np.any(original[795:912,790:916] != 255, axis=2)
shape = np.zeros_like(question)
shape[817:918,791:921] = np.any(target[817:918,791:921] != 255, axis=2)
# Order the copied stroke clockwise around its contour.
vertices = np.array([[852,834],[880,824],[909,852],[852,909],[796,852],[824,824],[852,834]],float)
y,x = np.where(shape)
p = np.column_stack((x,y))
best = np.full(len(x),np.inf)
progress = np.zeros(len(x))
length = 0.
for a,b in zip(vertices[:-1],vertices[1:]):
    v = b-a
    seglen = np.linalg.norm(v)
    t = np.clip(((p-a)@v)/(seglen**2),0,1)
    d = np.linalg.norm(p-(a+t[:,None]*v),axis=1)
    select = d < best
    progress[select] = length+t[select]*seglen
    best[select] = d[select]
    length += seglen
cmd = ['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-preset','slow','-qp','0','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'video.mp4')]
proc = subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
for i in range(35):
    frame = original.copy()
    if i:
        fade = min(i/9,1)
        frame[question] = np.rint(original[question]*(1-fade)+255*fade).astype(np.uint8)
    if i >= 10:
        selected = progress <= length*min((i-9)/24,1)
        frame[y[selected],x[selected]] = target[y[selected],x[selected]]
    if i == 34:
        assert np.array_equal(frame,target)
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
err = proc.stderr.read()
if proc.wait():
    raise RuntimeError(err.decode())
