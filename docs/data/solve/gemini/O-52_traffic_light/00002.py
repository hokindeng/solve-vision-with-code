import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import subprocess
import os

def get_state(t):
    t = t % 16
    if t < 4:
        return 'Red', 4 - t
    elif t < 8:
        return 'Yellow', 8 - t
    elif t < 12:
        return 'Green', 12 - t
    else:
        return 'Yellow', 16 - t

def color_to_bgr(color):
    if color == 'Red':
        return [0, 0, 255]
    elif color == 'Yellow':
        return [0, 200, 255]
    elif color == 'Green':
        return [0, 200, 0]

# Load first frame
img = cv2.imread('/app/first_frame.png')

# Extract digits
img1 = img[568:691, 743:866].copy()
img3 = img[276:399, 451:574].copy()
img4 = img[860:983, 451:574].copy()

# Generate 2
font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 108)
pil_img2 = Image.new('RGB', (123, 123), 'white')
draw = ImageDraw.Draw(pil_img2)
bbox = draw.textbbox((0, 0), '2', font=font)
w = bbox[2] - bbox[0]
h = bbox[3] - bbox[1]
draw.text(((123-w)/2 - bbox[0], (123-h)/2 - bbox[1]), '2', font=font, fill='black')
img2 = np.array(pil_img2)
# PIL is RGB, OpenCV is BGR, but since it's black and white, it's the same.

digits = {
    1: img1,
    2: img2,
    3: img3,
    4: img4
}

# Extract circle mask (from North Red)
circle_mask = (img[155:274, 447:578, 2] == 255) & (img[155:274, 447:578, 1] == 0) & (img[155:274, 447:578, 0] == 0)

# Define locations
lights = {
    'N': {'light': (155, 274, 447, 578), 'text': (276, 399, 451, 574), 't0': 1},
    'S': {'light': (739, 858, 447, 578), 'text': (860, 983, 451, 574), 't0': 4},
    'W': {'light': (447, 566, 155, 286), 'text': (568, 691, 159, 282), 't0': 11},
    'E': {'light': (447, 566, 739, 870), 'text': (568, 691, 743, 866), 't0': 11}
}

os.makedirs('/app/frames', exist_ok=True)

# Generate 80 frames
for frame_idx in range(80):
    # Time in seconds
    s = frame_idx // 16
    
    frame = img.copy()
    
    for key, info in lights.items():
        t = info['t0'] + s
        color, count = get_state(t)
        
        # Apply color
        y1, y2, x1, x2 = info['light']
        bgr = color_to_bgr(color)
        
        # We must overwrite the light color.
        # But wait, first we need to make sure we clear the OLD color!
        # Since the background of the light is NOT changing, the circle_mask is exactly the colored part.
        # We can just set the pixels inside the mask to the new color.
        # Because every light has the exact same mask shape!
        roi = frame[y1:y2, x1:x2]
        roi[circle_mask] = bgr
        frame[y1:y2, x1:x2] = roi
        
        # Apply text
        ty1, ty2, tx1, tx2 = info['text']
        frame[ty1:ty2, tx1:tx2] = digits[count]
        
    cv2.imwrite(f'/app/frames/frame_{frame_idx:04d}.png', frame)

# Encode to video
# Requirements: H.264, yuv420p, 1024x1024, 16 fps.
os.makedirs('/app/output', exist_ok=True)
cmd = [
    'ffmpeg', '-y', '-framerate', '16', '-i', '/app/frames/frame_%04d.png',
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-s', '1024x1024',
    '/app/output/video.mp4'
]
subprocess.run(cmd, check=True)

