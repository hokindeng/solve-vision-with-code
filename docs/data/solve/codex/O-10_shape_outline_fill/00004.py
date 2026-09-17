from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

ROOT = Path('/app')

def smooth(x):
    x = np.clip(x, 0, 1)
    return x*x*(3-2*x)

def main():
    original = np.array(Image.open(ROOT/'first_frame.png').convert('RGB'))
    h,w = original.shape[:2]
    yy,xx = np.indices((h,w))
    # Move an exact copy of the lower source circle into the answer position.
    source = np.full_like(original,255)
    source[682:856, 776:952] = original[682:856,72:248]
    circle = np.any(source != 255,axis=2)
    question = (xx >= 840)&(xx <= 887)&(yy >= 739)&(yy <= 798)&np.any(original!=255,axis=2)
    angle = np.mod(np.arctan2(yy-768.5,xx-864),2*np.pi)
    # Short, regularly spaced green dashes, as in the upper answer.
    phase = np.mod(angle/(2*np.pi)*62,1)
    dash = phase < .61
    out = ROOT/'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r','16','-i','-','-an','-c:v','libx264','-preset','slow','-crf','0','-pix_fmt','yuv420p',str(out/'video.mp4')],stdin=subprocess.PIPE)
    for i in range(60):
        frame = original.copy()
        erase = smooth(i/12)
        frame[question] = np.rint(original[question]*(1-erase)+255*erase).astype(np.uint8)
        reveal = smooth((i-10)/17)
        # The emerging solid circle gradually takes the dashed outline style.
        change = smooth((i-27)/27)
        alpha = np.where(dash, reveal, reveal*(1-change))
        frame[circle] = np.rint(255+(source[circle].astype(float)-255)*alpha[circle,None]).astype(np.uint8)
        if i == 0:
            assert np.array_equal(frame,original)
        assert np.array_equal(frame[~(circle|question)],original[~(circle|question)])
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
