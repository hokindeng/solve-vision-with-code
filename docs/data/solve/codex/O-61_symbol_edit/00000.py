from PIL import Image
import numpy as np
import subprocess
from pathlib import Path

ROOT=Path('/app')
source=np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
# Extract the original artwork, including its original antialiasing.
xs=[44+105*i for i in range(9)]
sprites={}
for name,i in [('circle',0),('diamond',1),('triangle',2),('square',3)]:
    sprites[name]=source[466:559,xs[i]+2:xs[i]+95].copy()
base=source.copy()
for i in range(4):
    base[466:559,xs[i]+2:xs[i]+95]=255

def render(sequence):
    frame=base.copy()
    for i,name in enumerate(sequence):
        if name:
            frame[466:559,xs[i]+2:xs[i]+95]=sprites[name]
    return frame

def mix(a,b,t):
    t=t*t*(3-2*t)
    return np.rint(a.astype(float)*(1-t)+b.astype(float)*t).astype(np.uint8)

initial=['circle','diamond','triangle','square']
stages=[
    ([None]+initial,['circle']+initial),
    (['circle','circle','diamond','triangle',None,'square'],['circle','circle','diamond','triangle','circle','square']),
    (['circle','circle','diamond','triangle','circle',None,'square'],['circle','circle','diamond','triangle','circle','circle','square']),
]
frames=[source.copy() for _ in range(6)]
previous=source
for shifted,inserted in stages:
    mid=render(shifted)
    end=render(inserted)
    for k in range(10):
        frames.append(mix(previous,mid,(k+1)/10))
    for k in range(12):
        frames.append(mix(mid,end,(k+1)/12))
    previous=end
frames.extend([previous.copy() for _ in range(4)])
assert len(frames)==76
assert np.array_equal(frames[0],source)
(ROOT/'output').mkdir(exist_ok=True)
process=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size','1024x1024','-framerate','16','-i','-','-an','-c:v','libx264','-crf','12','-preset','slow','-pix_fmt','yuv420p','-movflags','+faststart',str(ROOT/'output/video.mp4')],stdin=subprocess.PIPE)
for frame in frames:
    process.stdin.write(frame.tobytes())
process.stdin.close()
if process.wait():
    raise RuntimeError('ffmpeg failed')
