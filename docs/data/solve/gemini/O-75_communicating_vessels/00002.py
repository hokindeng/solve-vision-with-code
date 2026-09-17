import numpy as np
from PIL import Image
import os
import subprocess

first_frame = Image.open('/app/first_frame.png')
first_frame_arr = np.array(first_frame)

tubes_cols = [(106, 195), (340, 429), (574, 663), (808, 897)]
y0 = [289, 549, 684, 724]
y_final = 564

num_frames = 53
fps = 16

os.makedirs('/app/output', exist_ok=True)

cmd = [
    'ffmpeg', '-y',
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-s', '1024x1024',
    '-pix_fmt', 'rgb24',
    '-r', str(fps),
    '-i', '-',
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    '/app/output/video.mp4'
]

process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

for f in range(num_frames):
    t = f / fps
    current_frame = np.copy(first_frame_arr)
    
    for i, (cmin, cmax) in enumerate(tubes_cols):
        y_float = y_final + (y0[i] - y_final) * np.exp(-2.29 * t)
        y_int = int(round(y_float))
        
        current_frame[138:y_int, cmin:cmax+1] = [255, 255, 255]
        current_frame[y_int:880, cmin:cmax+1] = [255, 227, 75]
        
    process.stdin.write(current_frame.tobytes())

process.stdin.close()
process.wait()
if process.returncode != 0:
    print("Error during video generation.")
else:
    print("Video generated successfully.")
