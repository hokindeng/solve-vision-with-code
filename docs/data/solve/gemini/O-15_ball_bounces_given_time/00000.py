import cv2
import numpy as np
import subprocess
import os

def generate_video():
    img = cv2.imread('/app/first_frame.png')
    
    walls = np.all(img == [100, 100, 100], axis=-1)
    arrow = np.all(img == [0, 140, 255], axis=-1)
    
    clean_bg = np.full((1024, 1024, 3), 255, dtype=np.uint8)
    clean_bg[walls] = [100, 100, 100]
    
    p0 = np.array([821.0, 819.0])
    p1 = np.array([747.528, 931.0])
    p2 = np.array([197.8, 93.0])
    
    d1 = np.linalg.norm(p1 - p0)
    d2 = np.linalg.norm(p2 - p1)
    total_d = d1 + d2
    
    num_frames = 79
    intervals = num_frames - 1
    
    os.makedirs('/app/output', exist_ok=True)
    out_path = '/app/output/video.mp4'
    
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', '1024x1024',
        '-pix_fmt', 'bgr24',
        '-r', '16',
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        out_path
    ]
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    
    for i in range(num_frames):
        if i == 0:
            frame = img.copy()
        else:
            dist = (i / intervals) * total_d
            if dist <= d1:
                t = dist / d1
                pos = p0 + t * (p1 - p0)
            else:
                t = (dist - d1) / d2
                pos = p1 + t * (p2 - p1)
            
            frame = clean_bg.copy()
            cv2.circle(frame, (int(round(pos[0])), int(round(pos[1]))), 30, (139, 61, 72), -1, cv2.LINE_AA)
            frame[arrow] = [0, 140, 255]
            
        process.stdin.write(frame.tobytes())
        
    process.stdin.close()
    process.wait()

if __name__ == '__main__':
    generate_video()
