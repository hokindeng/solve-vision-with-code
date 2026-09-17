import cv2
import numpy as np
import imageio
import os

os.makedirs('/app/output', exist_ok=True)

img = cv2.imread('first_frame.png')
# Convert to RGB for imageio
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

agent_crop = img_rgb[425:494, 221:290].copy()
orange_mask = np.all(agent_crop == [255, 165, 0], axis=-1)

bg_img = img_rgb.copy()
bg_img[425:494, 221:290][orange_mask] = [0, 255, 0]

def get_tl(c, r):
    return (c * 102 + 17, r * 102 + 17)

W0 = get_tl(2, 4)
W1 = get_tl(2, 9)
W2 = get_tl(6, 9)
W3 = get_tl(3, 9)
W4 = get_tl(3, 2)
W5 = get_tl(6, 2)
W6 = get_tl(6, 5)
W7 = get_tl(9, 5)
W8 = get_tl(9, 2)

waypoints = [W0, W1, W2, W3, W4, W5, W6, W7, W8]

def dist(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

total_dist = sum(dist(waypoints[i], waypoints[i+1]) for i in range(len(waypoints)-1))

frames = []
for i in range(113):
    d = i * total_dist / 112.0
    
    curr_d = 0
    for j in range(len(waypoints)-1):
        seg_dist = dist(waypoints[j], waypoints[j+1])
        if curr_d + seg_dist >= d - 1e-5:
            t = (d - curr_d) / seg_dist if seg_dist > 0 else 0
            x = waypoints[j][0] + t * (waypoints[j+1][0] - waypoints[j][0])
            y = waypoints[j][1] + t * (waypoints[j+1][1] - waypoints[j][1])
            break
        curr_d += seg_dist
    else:
        x, y = waypoints[-1]
        
    x, y = int(round(x)), int(round(y))
    
    frame = bg_img.copy()
    for c in range(3):
        frame[y:y+69, x:x+69, c] = np.where(orange_mask, agent_crop[:, :, c], frame[y:y+69, x:x+69, c])
        
    frames.append(frame)

imageio.mimwrite('/app/output/video.mp4', frames, fps=16, macro_block_size=None, codec='libx264', pixelformat='yuv420p')
