import cv2
import numpy as np
import os
import subprocess

def solve():
    os.makedirs('/app/output', exist_ok=True)
    img = cv2.imread('/app/first_frame.png')
    
    dominos = [
        {"id": 0, "x": 60, "y": 447, "w": 55, "h": 123, "step": 0, "last": False},
        {"id": 1, "x": 201, "y": 447, "w": 55, "h": 123, "step": 1, "last": False},
        {"id": 2, "x": 342, "y": 447, "w": 55, "h": 123, "step": 2, "last": False},
        {"id": 3, "x": 483, "y": 538, "w": 55, "h": 123, "step": 3, "last": False},
        {"id": 4, "x": 483, "y": 356, "w": 55, "h": 123, "step": 3, "last": False},
        {"id": 5, "x": 624, "y": 556, "w": 55, "h": 123, "step": 4, "last": True},
        {"id": 6, "x": 624, "y": 337, "w": 55, "h": 123, "step": 4, "last": False},
        {"id": 7, "x": 765, "y": 319, "w": 55, "h": 123, "step": 5, "last": True},
    ]

    patches = []
    bg = img.copy()

    for d in dominos:
        x, y, w, h = d["x"], d["y"], d["w"], d["h"]
        patch = img[y:y+h, x:x+w].copy()
        diff = np.abs(patch.astype(np.int32) - np.array([252, 252, 252]))
        alpha = np.clip(np.sum(diff, axis=-1) / 10.0, 0, 1.0)
        patches.append({"patch": patch, "alpha": alpha, "d": d})
        bg[y:y+h, x:x+w] = [252, 252, 252]

    def get_angle(t, is_last):
        if t <= 0:
            return 0
        elif t < 6:
            return 45 * (t / 6.0) ** 1.2
        else:
            target = 85 if is_last else 65
            if t >= 18:
                return target
            progress = (t - 6) / 12.0
            return 45 + (target - 45) * (progress ** 0.8)

    patches.sort(key=lambda p: p["d"]["x"], reverse=True)

    frames_dir = '/tmp/frames_final'
    os.makedirs(frames_dir, exist_ok=True)

    for frame_idx in range(54):
        if frame_idx == 0:
            frame = img.copy()
        else:
            frame = bg.copy()
            for p in patches:
                d = p["d"]
                t = frame_idx - d["step"] * 6
                angle = get_angle(t, d["last"])
                
                if angle == 0:
                    x, y, w, h = d["x"], d["y"], d["w"], d["h"]
                    roi = frame[y:y+h, x:x+w]
                    alpha_src = p["alpha"][:, :, np.newaxis]
                    frame[y:y+h, x:x+w] = roi * (1 - alpha_src) + p["patch"] * alpha_src
                else:
                    x, y, w, h = d["x"], d["y"], d["w"], d["h"]
                    pivot_x, pivot_y = x + w, y + h
                    
                    canvas_size = 300
                    cx, cy = 150, 150
                    
                    canvas_rgb = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
                    canvas_a = np.zeros((canvas_size, canvas_size), dtype=np.float32)
                    
                    px = cx - w
                    py = cy - h
                    canvas_rgb[py:py+h, px:px+w] = p["patch"]
                    canvas_a[py:py+h, px:px+w] = p["alpha"]
                    
                    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
                    rot_rgb = cv2.warpAffine(canvas_rgb, M, (canvas_size, canvas_size), flags=cv2.INTER_LINEAR)
                    rot_a = cv2.warpAffine(canvas_a, M, (canvas_size, canvas_size), flags=cv2.INTER_LINEAR)
                    
                    top_left_x = pivot_x - cx
                    top_left_y = pivot_y - cy
                    
                    y1_dst = max(0, top_left_y)
                    y2_dst = min(1024, top_left_y + canvas_size)
                    x1_dst = max(0, top_left_x)
                    x2_dst = min(1024, top_left_x + canvas_size)
                    
                    y1_src = max(0, -top_left_y)
                    y2_src = y1_src + (y2_dst - y1_dst)
                    x1_src = max(0, -top_left_x)
                    x2_src = x1_src + (x2_dst - x1_dst)
                    
                    if y1_dst < y2_dst and x1_dst < x2_dst:
                        roi_dst = frame[y1_dst:y2_dst, x1_dst:x2_dst]
                        roi_src = rot_rgb[y1_src:y2_src, x1_src:x2_src]
                        alpha_src = rot_a[y1_src:y2_src, x1_src:x2_src, np.newaxis]
                        
                        frame[y1_dst:y2_dst, x1_dst:x2_dst] = roi_dst * (1 - alpha_src) + roi_src * alpha_src
                        
        cv2.imwrite(f'{frames_dir}/frame_{frame_idx:03d}.png', frame)

    cmd = [
        'ffmpeg', '-y', '-framerate', '16',
        '-i', f'{frames_dir}/frame_%03d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '/app/output/video.mp4'
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == '__main__':
    solve()
