import cv2, numpy as np
from scipy.optimize import least_squares
from PIL import Image
import subprocess, os

ROOT='/app'
im=np.array(Image.open(ROOT+'/first_frame.png').convert('RGB'))
mask=(im.min(2)<250)&(np.indices(im.shape[:2])[1]<470)
n,labels,stats,centers=cv2.connectedComponentsWithStats(mask.astype(np.uint8))
objects=[]
# Expected target geometry supplies a region for fitting its dashed boundary.
targets=[np.array(p,float) for p in [
 [[636,275],[725,336],[714,441],[619,484],[531,422],[541,319]],
 [[732,336],[882,276],[944,424],[794,484]],
 [[863,493],[914,666],[741,623]],
 [[551,542],[657,495],[705,601],[598,647]]]]
yy,xx=np.where((im[:,:,0]<180)&(im[:,:,1]<180)&(im[:,:,2]<180)&(np.indices(im.shape[:2])[1]>490))
points=np.stack([xx,yy],1).astype(float)
def edge_distance(p,poly):
 a=poly; b=np.roll(poly,-1,axis=0); v=b-a
 t=np.clip(((p[:,None,:]-a)*v).sum(2)/(v*v).sum(1),0,1)
 return np.sqrt(((p[:,None,:]-a-t[:,:,None]*v)**2).sum(2)).min(1)
initial=[(-9,475),(-27,482),(18,517),(-35,483)]
for i in range(1,n):
 if stats[i,4]<100: continue
 x,y,w,h,area=stats[i]; center=centers[i]
 full=(labels==i).astype(np.uint8)
 contour=cv2.findContours(full,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0][0]
 poly=cv2.approxPolyDP(contour,1.3,True).reshape(-1,2).astype(float)
 idx=len(objects)
 if idx<4:
  pts=points[edge_distance(points,targets[idx])<4]
  def transform(par):
   a=np.deg2rad(par[0]); rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
   return (poly-center)@rot.T+center+[par[1],0]
  fit=least_squares(lambda p:edge_distance(pts,transform(p)),initial[idx],diff_step=0.0001)
  angle,dx=fit.x
 else: angle,dx=0,500
 print(idx,'center',center,'angle',angle,'dx',dx,flush=True)
 # Full-size alpha retains the original fine gray shape borders.
 objects.append((full,center,angle,dx))
base=im.copy();base[mask]=255
os.makedirs(ROOT+'/output',exist_ok=True)
proc=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',ROOT+'/output/video.mp4'],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
def ease(t):
 t=np.clip(t,0,1);return t*t*(3-2*t)
for frame in range(48):
 if frame==0: out=im
 else:
  t=frame/47
  rot_t=ease(t/0.42); move_t=ease((t-0.42)/0.58)
  canvas=base.astype(np.float32)
  for full,center,angle,dx in objects:
   a=np.deg2rad(angle*rot_t); r=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
   mat=np.column_stack([r,center-r@center+np.array([dx*move_t,0])])
   alpha=cv2.warpAffine(full.astype(np.float32),mat,(1024,1024),flags=cv2.INTER_CUBIC).clip(0,1)
   # Warp distance from white, so interpolation at the boundary has no dark fringe.
   ink=(255-im.astype(np.float32))*full[:,:,None]
   moved=cv2.warpAffine(ink,mat,(1024,1024),flags=cv2.INTER_CUBIC)
   canvas=canvas*(1-alpha[:,:,None])+255*alpha[:,:,None]-moved
  out=np.clip(canvas,0,255).astype(np.uint8)
 proc.stdin.write(out.tobytes())
 if frame==47: Image.fromarray(out).save(ROOT+'/output/last_frame.png')
proc.stdin.close();proc.wait()
assert proc.returncode==0
