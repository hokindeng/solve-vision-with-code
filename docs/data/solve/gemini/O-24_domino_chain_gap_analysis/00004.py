from PIL import Image
import numpy as np
import imageio
import os

def get_angles(f, num_frames=50):
    t = f / max(1, (num_frames - 1))
    t = t ** 1.5 
    max_a = [41.0, 53.0, 90.0]
    start_t = [0.0, 0.15, 0.30]
    angles = [0.0, 0.0, 0.0]
    for i in range(3):
        if t <= start_t[i]:
            angles[i] = 0.0
        else:
            prog = (t - start_t[i]) / (1.0 - start_t[i])
            angles[i] = max_a[i] * (prog ** 1.2)
    return angles

def main():
    img = Image.open('/app/first_frame.png').convert('RGBA')
    bg = img.copy()

    # Domino x ranges: from 145 to 191 (exclusive), etc.
    dom_x = [ (145, 191), (204, 250), (265, 311) ]
    
    dom_images = []
    
    for start_x, end_x in dom_x:
        dom = img.crop((start_x, 588, end_x, 722))
        dom_images.append(dom)
        
        for x in range(start_x, end_x):
            for y in range(588, 720):
                bg.putpixel((x, y), (255, 255, 255, 255))
            for y in range(720, 722):
                bg.putpixel((x, y), (85, 115, 139, 255))
                
    pivots = [
        (190, 721),
        (249, 721),
        (310, 721)
    ]
    
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    num_frames = 50
    for f in range(num_frames):
        frame = bg.copy()
        angles = get_angles(f, num_frames)
        
        for i in range(2, -1, -1):
            if angles[i] > 0:
                layer = Image.new('RGBA', (1024, 1024), (0, 0, 0, 0))
                layer.paste(dom_images[i], (dom_x[i][0], 588))
                layer = layer.rotate(-angles[i], center=pivots[i], resample=Image.Resampling.BICUBIC)
                frame = Image.alpha_composite(frame, layer)
            else:
                frame.paste(dom_images[i], (dom_x[i][0], 588), dom_images[i])
                
        writer.append_data(np.array(frame.convert('RGB')))
        
    writer.close()

if __name__ == '__main__':
    main()
