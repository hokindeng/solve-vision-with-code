import numpy as np
from PIL import Image, ImageDraw
import cv2
from scipy.optimize import least_squares, minimize_scalar
import subprocess, os

ROOT='/app'
im=np.array(Image.open(ROOT+'/first_frame.png').convert('RGB'))
# Recover the six straight polygon edges from their original raster locations.
vertices=np.array([[793.,214.],[812.,157.],[897.,186.],[907.,158.],[964.,177.],[934.,262.]])
center=np.array([793.,214.])
y,x=np.where(np.all(im==[100,100,100],axis=2))
target_pixels=np.column_stack([x,y])
def rotate(angle):
    a=np.deg2rad(angle)
    R=np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]])
    return (vertices-center)@R.T+center

def distances(points,poly):
    out=[]
    for a,b in zip(poly,np.roll(poly,-1,axis=0)):
        v=b-a
        t=np.clip((points-a)@v/(v@v),0,1)
        out.append(np.sum((points-a-t[:,None]*v)**2,axis=1))
    return np.min(out,axis=0)
angle=minimize_scalar(lambda a:np.mean(distances(target_pixels,rotate(a))),bounds=(72,79),method='bounded').x
# Clear only the original polygon; retain all target dashes and the marker.
base=im.copy()
poly_mask=np.zeros(im.shape[:2],np.uint8)
cv2.fillPoly(poly_mask,[vertices.astype(np.int32)],255)
base[poly_mask>0]=[240,240,240]
# Restore the circular center marker exactly, including its black border.
yy,xx=np.indices(im.shape[:2])
marker=((xx-793)**2+(yy-214)**2<=13**2)
marker &= np.all(im[:,:,0:1]==im,axis=2) & (~np.all(im==[240,240,240],axis=2))
# Preserve the marker's exact source patch while leaving outside pixels untouched.
marker_region=(xx>=781)&(xx<=805)&(yy>=202)&(yy<=226)
marker_region &= ((xx-793)**2+(yy-214)**2<=12.7**2)
# Its circular pixels are black and white; background outside the disk stays dynamic.
marker_region &= (np.all(im==[0,0,0],axis=2)|np.all(im==[255,255,255],axis=2))
os.makedirs(ROOT+'/output',exist_ok=True)
proc=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',ROOT+'/output/video.mp4'],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
for i in range(70):
    if i==0:
        frame=im.copy()
    else:
        t=i/69
        t=t*t*(3-2*t)
        poly=rotate(angle*t)
        img=Image.fromarray(base)
        draw=ImageDraw.Draw(img)
        coords=[tuple(p) for p in poly]
        draw.polygon(coords, fill=(60,139,119))
        draw.line(coords+[coords[0]],fill=(50,50,50),width=1)
        frame=np.array(img)
        frame[marker_region]=im[marker_region]
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
assert proc.wait()==0
print('Rotation:',angle)
