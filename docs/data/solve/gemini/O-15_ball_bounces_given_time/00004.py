import cv2
import numpy as np
import math
import os
import imageio

def main():
    img = cv2.imread('/app/first_frame.png')
    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask_1 = (img == [255, 140, 0]).all(axis=2) # Arrow is [0, 140, 255] in BGR -> [255, 140, 0] in RGB
    mask_2 = (img == [138, 43, 226]).all(axis=2) # Ball is [226, 43, 138] in BGR -> [138, 43, 226] in RGB

    ball_patch = mask_2[405:466, 702:763].copy()
    patch_flip_x = np.fliplr(ball_patch)
    patch_flip_y = np.flipud(ball_patch)
    patch_flip_xy = np.fliplr(np.flipud(ball_patch))
    full_ball = ball_patch | patch_flip_x | patch_flip_y | patch_flip_xy

    bg = img.copy()
    bg[405:466, 702:763][full_ball] = [255, 255, 255]
    # restore arrow in bg just in case, wait, if bg is used as base, we don't restore arrow into bg
    # but rather we draw arrow on top of each frame.
    # Actually, bg should NOT have the arrow where it overlaps the initial ball, wait.
    # The arrow is on the white background. If we do `bg[...][full_ball] = [255, 255, 255]`, 
    # it erases the ball AND the arrow over the ball.
    # We should put the arrow back into `bg` so it's always there!
    bg[mask_1] = [255, 140, 0]

    cx, cy = 732.0, 435.0
    vx, vy = 150.0, -58.0
    length = math.hypot(vx, vy)
    vx /= length
    vy /= length

    min_x, max_x = 93.0, 931.0
    min_y, max_y = 93.0, 931.0

    pts = [(cx, cy)]
    curr_cx, curr_cy = cx, cy

    for _ in range(6):
        tx = float('inf')
        if vx > 0: tx = (max_x - curr_cx) / vx
        elif vx < 0: tx = (min_x - curr_cx) / vx
            
        ty = float('inf')
        if vy > 0: ty = (max_y - curr_cy) / vy
        elif vy < 0: ty = (min_y - curr_cy) / vy
            
        t = min(tx, ty)
        curr_cx = curr_cx + vx * t
        curr_cy = curr_cy + vy * t
        
        if abs(t - tx) < 1e-9: vx = -vx
        if abs(t - ty) < 1e-9: vy = -vy

        pts.append((curr_cx, curr_cy))

    dists = []
    for i in range(len(pts)-1):
        d = math.hypot(pts[i+1][0] - pts[i][0], pts[i+1][1] - pts[i][1])
        dists.append(d)

    num_frames = 80
    total_dist = sum(dists)
    speed = total_dist / (num_frames - 1)

    frame_pts = []
    for i in range(num_frames):
        target_d = i * speed
        cur_d = 0
        for j in range(len(dists)):
            if cur_d + dists[j] >= target_d - 1e-9:
                rem = target_d - cur_d
                ratio = rem / dists[j]
                x = pts[j][0] + ratio * (pts[j+1][0] - pts[j][0])
                y = pts[j][1] + ratio * (pts[j+1][1] - pts[j][1])
                frame_pts.append((x, y))
                break
            cur_d += dists[j]

    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')

    for i, (fx, fy) in enumerate(frame_pts):
        frame = bg.copy()
        
        x = int(round(fx))
        y = int(round(fy))
        
        y0 = y - 30
        y1 = y + 31
        x0 = x - 30
        x1 = x + 31
        
        patch_in_frame = frame[y0:y1, x0:x1]
        patch_in_frame[full_ball] = [138, 43, 226]
        
        # Keep arrow strictly on top
        frame[mask_1] = [255, 140, 0]
        
        writer.append_data(frame)

    writer.close()

if __name__ == '__main__':
    main()
