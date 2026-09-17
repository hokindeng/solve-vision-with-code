import os, cv2, numpy as np
from PIL import Image
import imageio.v2 as imageio

ROOT='/app'
original=np.array(Image.open(ROOT+'/first_frame.png').convert('RGB'))
gray=cv2.cvtColor(original,cv2.COLOR_RGB2GRAY)
mask=np.uint8(gray<250); mask[:,480:]=0
n,labels,stats,cents=cv2.connectedComponentsWithStats(mask)
ids=[i for i in range(1,n) if stats[i,4]>100]
# Source polygon vertices, in clockwise order, and the stationary matching outlines.
sources=[None,[[380,291],[436,314],[443,374],[395,410],[340,386],[333,327]],[[90,502],[309,536],[274,756],[55,721]],[[412,566],[448,655],[359,691],[323,602]],[[382,699],[400,784],[317,757]]]
targets=[None,[[860,290],[912,320],[912,380],[860,410],[807,380],[807,320]],[[613,485],[818,568],[735,773],[530,690]],[[850,571],[944,595],[923,687],[828,664]],[[888,697],[930,773],[843,771]]]

def fitted_vertices(guess):
    p=np.asarray(guess,float)
    yy,xx=np.where((gray==90)&(np.indices(gray.shape)[1]>480))
    points=np.column_stack((xx,yy)).astype(float)
    lines=[]
    for a,b in zip(p,np.roll(p,-1,axis=0)):
        v=b-a; length=np.linalg.norm(v); v/=length
        along=(points-a)@v
        dist=np.abs(np.cross(v,points-a))
        pts=points[(dist<4)&(along>3)&(along<length-3)]
        mid=pts.mean(axis=0)
        _,_,vh=np.linalg.svd(pts-mid,full_matrices=False)
        normal=np.array([-vh[0,1],vh[0,0]])
        lines.append((normal,normal@mid))
    return np.array([np.linalg.solve(np.array([lines[j-1][0],lines[j][0]]),np.array([lines[j-1][1],lines[j][1]])) for j in range(len(p))])

base=original.copy(); base[labels>0]=255
objects=[]
for k,i in enumerate(ids):
    isolated=np.full_like(original,255); isolated[labels==i]=original[labels==i]
    if k==0:
        center=np.array([186.,350.]); dx=488.; angle=0.
    else:
        src=np.array(sources[k],float); dst=fitted_vertices(targets[k])
        center=src.mean(axis=0); a=src-center; b=dst-dst.mean(axis=0)
        angle=np.arctan2(np.sum(a[:,0]*b[:,1]-a[:,1]*b[:,0]),np.sum(a*b))
        dx=dst.mean(axis=0)[0]-center[0]
        print(k,'center',center,'target',dst.mean(axis=0),'angle',np.degrees(angle),'dx',dx)
    objects.append((isolated,center,dx,angle))

os.makedirs(ROOT+'/output',exist_ok=True)
writer=imageio.get_writer(ROOT+'/output/video.mp4',fps=16,codec='libx264',pixelformat='yuv420p',quality=9,macro_block_size=None,ffmpeg_params=['-crf','16'])
# Ease each phase, using all 48 frames. All five objects share phase timing.
def ease(t):
    t=np.clip(t,0,1); return t*t*(3-2*t)
for frame in range(48):
    if frame==0:
        canvas=original.copy()
    else:
        rotation=ease(frame/19)
        movement=ease((frame-19)/28)
        canvas=base.copy()
        for sprite,center,dx,angle in objects:
            theta=angle*rotation
            c,s=np.cos(theta),np.sin(theta)
            R=np.array([[c,-s],[s,c]])
            shift=center-R@center+np.array([dx*movement,0])
            M=np.column_stack((R,shift))
            moved=cv2.warpAffine(sprite,M,(1024,1024),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
            region=np.any(moved<255,axis=2)
            canvas[region]=moved[region]
        # Keep every pixel of each stationary dashed outline visible and unchanged.
        stationary=np.any(base<255,axis=2)
        canvas[stationary]=base[stationary]
    writer.append_data(canvas)
writer.close()
Image.fromarray(canvas).save(ROOT+'/output/last_frame.png')
