import os
import cv2
import numpy as np
import subprocess

os.makedirs('/app/output', exist_ok=True)
os.makedirs('/app/frames', exist_ok=True)

img = cv2.imread('/app/first_frame.png')

# Cell 0,1 (Black Heart)
cell01 = img[0:341, 341:682].astype(np.float32)
# Cell 2,2 (Question Mark)
original_cell22 = img[682:1024, 682:1024].astype(np.float32)

# 1. Clean the background of Cell 2,2 by removing the question mark
qm_binary = (cv2.cvtColor(original_cell22.astype(np.uint8), cv2.COLOR_BGR2GRAY) < 200).astype(np.uint8) * 255
qm_binary[:5, :] = 0; qm_binary[-5:, :] = 0; qm_binary[:, :5] = 0; qm_binary[:, -5:] = 0
clean_cell22 = cv2.inpaint(original_cell22.astype(np.uint8), qm_binary, 3, cv2.INPAINT_TELEA).astype(np.float32)

# 2. Extract the heart's alpha mask from Cell 0,1
heart_binary = (cv2.cvtColor(cell01.astype(np.uint8), cv2.COLOR_BGR2GRAY) < 200).astype(np.uint8) * 255
heart_binary[:5, :] = 0; heart_binary[-5:, :] = 0; heart_binary[:, :5] = 0; heart_binary[:, -5:] = 0
bg_01 = cv2.inpaint(cell01.astype(np.uint8), heart_binary, 3, cv2.INPAINT_TELEA).astype(np.float32)

diff = bg_01 - cell01
alpha = np.mean(diff / (bg_01 + 1e-5), axis=2)
alpha = np.clip(alpha, 0, 1)

# Pad alpha to 342x342 to align perfectly with the target cell
alpha_342 = np.pad(alpha, ((0, 1), (0, 1)), mode='constant', constant_values=0)
alpha_342 = alpha_342[:, :, np.newaxis]

# 3. Create the final destination cell (Black Heart on Cell 2,2 background)
heart_color = np.array([0, 0, 0], dtype=np.float32)
final_cell22 = clean_cell22 * (1 - alpha_342) + heart_color * alpha_342

# 4. Generate 35 frames for the animation
num_frames = 35
for i in range(num_frames):
    progress = i / max(1, (num_frames - 1))
    
    # Wipe transition left-to-right to "draw" it in
    wipe_center = 80 + progress * 180
    x_grid = np.arange(342).reshape(1, 342)
    transition_mask = np.clip((wipe_center - x_grid) / 20.0, 0, 1)
    transition_mask = transition_mask[:, :, np.newaxis]
    
    frame_cell22 = original_cell22 * (1 - transition_mask) + final_cell22 * transition_mask
    
    frame = img.copy()
    frame[682:1024, 682:1024] = frame_cell22.astype(np.uint8)
    
    cv2.imwrite(f'/app/frames/frame_{i:04d}.png', frame)

# 5. Encode to mp4
subprocess.run([
    'ffmpeg', '-y', '-framerate', '16', '-i', '/app/frames/frame_%04d.png',
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
], check=True)
