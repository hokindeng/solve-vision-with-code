from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
FPS = 16

def main():
    base = np.array(Image.open(ROOT / 'first_frame.png').convert('RGB'))
    # Positions refer to the unchanged white interiors of the countdown boxes.
    positions = [(451,276), (451,860), (743,568), (159,568)]
    tiles = {}
    for digit, (x,y) in [('4',positions[0]), ('3',positions[2]), ('1',positions[3])]:
        tiles[int(digit)] = base[y:y+123,x:x+123].copy()
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',102)
    # Match the original text baseline.
    tile = Image.new('RGB',(123,123),'white')
    ImageDraw.Draw(tile).text((61.5-font.getlength('2')/2,3),'2', font=font, fill='black')
    tiles[2] = np.array(tile)
    colors = [(255,0,0), (255,204,0), (0,204,0), (255,204,0)]
    # Cycle offsets: red, yellow, green, yellow, each lasting four seconds.
    offsets = [0,8,9,7]
    centers = [(512,220),(512,804),(804,512),(220,512)]
    masks = []
    yy,xx = np.indices(base.shape[:2])
    for cx,cy in centers:
        region = (abs(xx-cx)<70)&(abs(yy-cy)<70)
        colored = (base[:,:,1] == 204) & (base[:,:,2] == 0)
        colored |= (base[:,:,0] == 255)&(base[:,:,1] == 0)&(base[:,:,2] == 0)
        masks.append(region & colored)
    out = ROOT/'output'
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1024','-r',str(FPS),'-i','-','-an','-c:v','libx264','-crf','15','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'video.mp4')],stdin=subprocess.PIPE)
    for i in range(112):
        elapsed = min(i//FPS,6)
        frame = base.copy()
        if elapsed:
            for (x,y),offset,mask in zip(positions,offsets,masks):
                phase = (offset+elapsed)%16
                frame[mask] = colors[phase//4]
                frame[y:y+123,x:x+123] = tiles[4-phase%4]
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('ffmpeg failed')

if __name__ == '__main__':
    main()
