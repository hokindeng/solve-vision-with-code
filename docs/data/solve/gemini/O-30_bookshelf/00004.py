import cv2
import numpy as np
import math
import os
import subprocess

def main():
    img = cv2.imread('/app/first_frame.png')
    bg_img = img.copy()

    moves = [
        (670, 68, 130),
        (700, 518, 225),
        (730, 158, 158),
        (760, 248, 158)
    ]

    books = {}
    for x, targ_x, h in moves:
        books[x] = img[512-h:512, x:x+25].copy()
        bg_img[512-h:512, x:x+25] = [255, 255, 255]

    os.makedirs('/app/output', exist_ok=True)

    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', '1024x1024', '-pix_fmt', 'bgr24', '-r', '16',
        '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '/app/output/video.mp4'
    ]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    for f in range(25):
        frame = bg_img.copy()
        for i, (start_x, targ_x, h) in enumerate(moves):
            if f == 0:
                curr_x = start_x
                curr_y = 512
            else:
                if f <= i * 6:
                    curr_x = start_x
                    curr_y = 512
                elif f > (i + 1) * 6:
                    curr_x = targ_x
                    curr_y = 512
                else:
                    step = f - i * 6
                    t = step / 6.0
                    t_smooth = 0.5 - 0.5 * math.cos(t * math.pi)
                    curr_x = int(start_x + (targ_x - start_x) * t_smooth)
                    curr_y = 512 - int(260 * math.sin(t * math.pi))
                    
            top_y = curr_y - h
            
            if top_y < 0:
                book_slice = books[start_x][-top_y:]
                frame[0:curr_y, curr_x:curr_x+25] = book_slice
            else:
                frame[top_y:curr_y, curr_x:curr_x+25] = books[start_x]
                
        process.stdin.write(frame.tobytes())

    process.stdin.close()
    process.wait()

if __name__ == '__main__':
    main()
