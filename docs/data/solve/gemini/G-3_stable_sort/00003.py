import cv2
import numpy as np
import os
import subprocess
import shutil

def solve():
    img = cv2.imread('/app/first_frame.png')
    bg_color = np.array([235, 235, 235])
    mask = np.any(img != bg_color, axis=-1).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    shapes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        shape_mask = (labels == i)
        colors = img[shape_mask]
        is_border = np.all(colors == [80, 80, 80], axis=-1)
        fill_colors = colors[~is_border]
        if len(fill_colors) > 0:
            unique_colors, counts = np.unique(fill_colors, axis=0, return_counts=True)
            main_color = tuple(unique_colors[np.argmax(counts)])
        else:
            main_color = (0, 0, 0)
        
        crop_img = img[y:y+h, x:x+w].copy()
        crop_mask = shape_mask[y:y+h, x:x+w].copy()
        
        shapes.append({
            'id': i,
            'x': x, 'y': y, 'w': w, 'h': h,
            'main_color': main_color,
            'crop_img': crop_img,
            'crop_mask': crop_mask
        })

    groups = {}
    for s in shapes:
        c = s['main_color']
        if c not in groups:
            groups[c] = []
        groups[c].append(s)

    group_keys = list(groups.keys())
    # Sort groups by their initial average horizontal position
    group_keys.sort(key=lambda k: np.mean([s['x'] for s in groups[k]]))

    ordered_shapes = []
    for k in group_keys:
        # Sort shapes within each group by their width (size)
        sorted_group = sorted(groups[k], key=lambda s: s['w'])
        ordered_shapes.extend(sorted_group)

    total_w = sum(s['w'] for s in ordered_shapes)
    gap = (1024 - total_w) / (len(ordered_shapes) + 1)

    current_x = gap
    for s in ordered_shapes:
        s['tx'] = int(round(current_x))
        s['ty'] = 512 - s['h'] // 2
        current_x += s['w'] + gap

    os.makedirs('/app/output', exist_ok=True)
    os.makedirs('/tmp/frames', exist_ok=True)

    num_frames = 96
    
    def ease_in_out(t):
        return 4 * t * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 3) / 2

    for frame_idx in range(num_frames):
        t = frame_idx / max(1, (num_frames - 1))
        progress = ease_in_out(t)
        
        frame = np.full_like(img, 235)
        
        for s in ordered_shapes:
            cur_x = int(round(s['x'] + (s['tx'] - s['x']) * progress))
            cur_y = int(round(s['y'] + (s['ty'] - s['y']) * progress))
            
            crop = s['crop_img']
            mask = s['crop_mask']
            
            target_region = frame[cur_y:cur_y+s['h'], cur_x:cur_x+s['w']]
            np.copyto(target_region, crop, where=mask[:, :, None])
            
        cv2.imwrite(f'/tmp/frames/frame_{frame_idx:04d}.png', frame)

    subprocess.run([
        'ffmpeg', '-y', '-framerate', '16', '-i', '/tmp/frames/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ], check=True)
    
    shutil.rmtree('/tmp/frames')

if __name__ == '__main__':
    solve()
