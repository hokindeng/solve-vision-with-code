import cv2
import numpy as np
import subprocess
import os
import math

# 1. Read first frame
img = cv2.imread('/app/first_frame.png')

# 2. Extract agent and create clean background
blue_mask = (img[:,:,0] == 255) & (img[:,:,1] == 0) & (img[:,:,2] == 0)
dark_blue_mask = (img[:,:,0] == 139) & (img[:,:,1] == 0) & (img[:,:,2] == 0)
agent_mask = blue_mask | dark_blue_mask

y, x = np.where(agent_mask)
ymin, ymax = y.min(), y.max()
xmin, xmax = x.min(), x.max()

agent_crop = img[ymin:ymax+1, xmin:xmax+1].copy()
mask_crop = agent_mask[ymin:ymax+1, xmin:xmax+1]

clean_bg = img.copy()
clean_bg[agent_mask] = [0, 128, 0]

# 3. Path coordinates
node4 = (644.0, 800.0)
node0 = (626.0, 286.0)
node1 = (322.0, 298.0)

dist1 = math.hypot(node0[0] - node4[0], node0[1] - node4[1])
dist2 = math.hypot(node1[0] - node0[0], node1[1] - node0[1])
total_dist = dist1 + dist2

# 4. Generate frames
os.makedirs('/app/output', exist_ok=True)
frames = []

def draw_agent(bg, cx, cy):
    h, w = agent_crop.shape[:2]
    tl_x = int(round(cx)) - (w - 1) // 2
    tl_y = int(round(cy)) - (h - 1) // 2
    
    frame = bg.copy()
    for i in range(h):
        for j in range(w):
            if mask_crop[i, j]:
                yy = tl_y + i
                xx = tl_x + j
                if 0 <= yy < frame.shape[0] and 0 <= xx < frame.shape[1]:
                    frame[yy, xx] = agent_crop[i, j]
    return frame

num_frames = 30
for i in range(num_frames):
    t = i / (num_frames - 1) # 0 to 1
    d = t * total_dist
    
    if d <= dist1:
        # on first edge
        ratio = d / dist1 if dist1 > 0 else 0
        cx = node4[0] + ratio * (node0[0] - node4[0])
        cy = node4[1] + ratio * (node0[1] - node4[1])
    else:
        # on second edge
        ratio = (d - dist1) / dist2 if dist2 > 0 else 0
        cx = node0[0] + ratio * (node1[0] - node0[0])
        cy = node0[1] + ratio * (node1[1] - node0[1])
        
    frames.append(draw_agent(clean_bg, cx, cy))

# Replace first frame with exact original image to satisfy "Its first frame is first_frame.png"
frames[0] = img.copy()

# Save frames to video
video_path = '/app/output/video.mp4'
h, w = frames[0].shape[:2]

cmd = [
    'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
    '-s', f'{w}x{h}', '-pix_fmt', 'bgr24', '-r', '16',
    '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
    video_path
]

process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for frame in frames:
    process.stdin.write(frame.tobytes())
process.stdin.close()
process.wait()
print(f"Video saved to {video_path}")
