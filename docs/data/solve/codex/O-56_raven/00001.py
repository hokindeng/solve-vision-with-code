from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')
def main():
    base = np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
    # Every row and column cycles square, triangle, diamond.
    # Reuse the original triangle pixels to preserve its precise appearance.
    sy, sx = np.where((base[:,:,1] > base[:,:,0]) & (base[:,:,1] > base[:,:,2]) & (np.indices(base.shape[:2])[0] < 300) & (np.indices(base.shape[:2])[1] > 400) & (np.indices(base.shape[:2])[1] < 620))
    colors = base[sy,sx].copy()
    dy, dx = sy+682, sx+341
    points = np.stack([sx,sy],axis=1)
    vertices = np.array([[511,114],[567,226],[455,226],[511,114]],float)
    distances = []
    phases = []
    offset = 0.
    for a,b in zip(vertices[:-1],vertices[1:]):
        v=b-a
        length=np.linalg.norm(v)
        t=np.clip((points-a)@v/(length*length),0,1)
        distances.append(np.linalg.norm(points-(a+t[:,None]*v),axis=1))
        phases.append(offset+t*length)
        offset+=length
    closest=np.argmin(distances,axis=0)
    phases=np.array(phases)[closest,np.arange(len(sx))]/offset
    question = np.zeros(base.shape[:2],bool)
    question[780:920,800:905] = np.any(base[780:920,800:905] != 255,axis=2)
    out=ROOT/'output'
    out.mkdir(exist_ok=True)
    proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-preset','slow','-crf','0','-pix_fmt','yuv420p',str(out/'video.mp4')],stdin=subprocess.PIPE)
    for i in range(35):
        frame=base.copy()
        fade=min(i/9,1)
        frame[question]=np.rint(base[question]*(1-fade)+255*fade).astype(np.uint8)
        if i>=10:
            visible=phases <= (i-10)/24
            frame[dy[visible],dx[visible]]=colors[visible]
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait()!=0:
        raise RuntimeError('ffmpeg failed')

if __name__=='__main__':
    main()
