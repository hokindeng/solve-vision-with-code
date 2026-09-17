import cv2
import numpy as np
import imageio
import os

def main():
    # 1. Read first frame and extract balls
    first_frame = cv2.imread('/app/first_frame.png')
    if first_frame is None:
        raise ValueError("Could not read /app/first_frame.png")
        
    # Known centers of the balls from the first frame
    centers = {
        'Red': [(468, 497), (521, 497), (468, 550)],
        'Blue': [(192, 337), (192, 390)],
        'Yellow': [(482, 783), (534, 783), (482, 835), (534, 835)],
        'Purple': [(218, 713), (271, 713), (218, 765), (271, 765), (218, 818)]
    }

    balls_info = []
    for color_name, pts in centers.items():
        for cx, cy in pts:
            x1, x2 = cx - 23, cx + 23
            y1, y2 = cy - 23, cy + 23
            patch = first_frame[y1:y2, x1:x2]
            
            # Mask out background and white pixels
            bg_mask = (patch[:,:,0] == 240) & (patch[:,:,1] == 240) & (patch[:,:,2] == 240)
            white_mask = (patch[:,:,0] == 255) & (patch[:,:,1] == 255) & (patch[:,:,2] == 255)
            ball_mask = ~(bg_mask | white_mask)
            
            balls_info.append({
                'color': color_name,
                'orig_cx': cx,
                'orig_cy': cy,
                'cx': cx,
                'cy': cy,
                'patch': patch.copy(),
                'mask': ball_mask.copy()
            })

    # 2. Create clean background
    clean_bg = first_frame.copy()
    for b in balls_info:
        cx, cy = b['orig_cx'], b['orig_cy']
        x1, x2 = cx - 23, cx + 23
        y1, y2 = cy - 23, cy + 23
        mask = b['mask']
        patch = clean_bg[y1:y2, x1:x2]
        patch[mask] = (240, 240, 240)
        clean_bg[y1:y2, x1:x2] = patch

    # Extract labels to keep them on top
    labels_mask = np.any(clean_bg != [240, 240, 240], axis=-1)

    # 3. Calculate movements using organic Katamari-style attachment
    def get_movement(cluster1, cluster2, target_dist=53.0):
        c1 = np.mean(cluster1, axis=0)
        c2 = np.mean(cluster2, axis=0)
        direction = c2 - c1
        dist = np.linalg.norm(direction)
        direction = direction / dist
        
        best_t = 0
        for t in np.linspace(0, dist, 1000):
            moved_c1 = cluster1 + direction * t
            dists = np.array([[np.linalg.norm(p1 - p2) for p2 in cluster2] for p1 in moved_c1])
            if np.min(dists) <= target_dist:
                best_t = t
                break
        return direction * best_t

    c_red = np.array(centers['Red'], dtype=float)
    c_blue = np.array(centers['Blue'], dtype=float)
    c_yellow = np.array(centers['Yellow'], dtype=float)
    c_purple = np.array(centers['Purple'], dtype=float)

    v1 = get_movement(c_red, c_blue)
    c1 = np.vstack((c_red + v1, c_blue))
    v2 = get_movement(c1, c_yellow)
    c2 = np.vstack((c1 + v2, c_yellow))
    v3 = get_movement(c2, c_purple)

    red_template = balls_info[0]

    def draw_frame(balls_state):
        frame = clean_bg.copy()
        for b in balls_state:
            cx, cy = int(round(b['cx'])), int(round(b['cy']))
            x1, x2 = cx - 23, cx + 23
            y1, y2 = cy - 23, cy + 23
            
            if x1 < 0 or y1 < 0 or x2 > 1024 or y2 > 1024:
                continue
                
            patch = frame[y1:y2, x1:x2]
            mask = b['mask']
            patch[mask] = b['patch'][mask]
            frame[y1:y2, x1:x2] = patch
            
        # Restore labels to be always on top
        frame[labels_mask] = clean_bg[labels_mask]
        return frame

    frames = []

    for frame_idx in range(80):
        current_state = []
        
        def add_ball(b_idx, offset_x, offset_y, force_red=False):
            b = balls_info[b_idx]
            new_b = {
                'cx': b['orig_cx'] + offset_x,
                'cy': b['orig_cy'] + offset_y,
                'mask': red_template['mask'] if force_red else b['mask'],
                'patch': red_template['patch'] if force_red else b['patch']
            }
            current_state.append(new_b)

        if frame_idx <= 15:
            # Phase 1: move to Blue
            progress = frame_idx / 15.0
            cur_v1 = v1 * progress
            for i in range(3): add_ball(i, cur_v1[0], cur_v1[1])
            for i in range(3, 14): add_ball(i, 0, 0)
                
        elif frame_idx <= 21:
            # Pause 1
            for i in range(3): add_ball(i, v1[0], v1[1], force_red=True)
            for i in range(3, 5): add_ball(i, 0, 0, force_red=(frame_idx >= 16))
            for i in range(5, 14): add_ball(i, 0, 0)
                
        elif frame_idx <= 42:
            # Phase 2: move to Yellow
            progress = (frame_idx - 22) / 20.0
            cur_v2 = v2 * progress
            for i in range(3): add_ball(i, v1[0] + cur_v2[0], v1[1] + cur_v2[1], force_red=True)
            for i in range(3, 5): add_ball(i, cur_v2[0], cur_v2[1], force_red=True)
            for i in range(5, 14): add_ball(i, 0, 0)
                
        elif frame_idx <= 48:
            # Pause 2
            for i in range(3): add_ball(i, v1[0] + v2[0], v1[1] + v2[1], force_red=True)
            for i in range(3, 5): add_ball(i, v2[0], v2[1], force_red=True)
            for i in range(5, 9): add_ball(i, 0, 0, force_red=(frame_idx >= 43))
            for i in range(9, 14): add_ball(i, 0, 0)
                
        elif frame_idx <= 69:
            # Phase 3: move to Purple
            progress = (frame_idx - 49) / 20.0
            cur_v3 = v3 * progress
            for i in range(3): add_ball(i, v1[0] + v2[0] + cur_v3[0], v1[1] + v2[1] + cur_v3[1], force_red=True)
            for i in range(3, 5): add_ball(i, v2[0] + cur_v3[0], v2[1] + cur_v3[1], force_red=True)
            for i in range(5, 9): add_ball(i, cur_v3[0], cur_v3[1], force_red=True)
            for i in range(9, 14): add_ball(i, 0, 0)
                
        else:
            # Pause 3
            for i in range(3): add_ball(i, v1[0] + v2[0] + v3[0], v1[1] + v2[1] + v3[1], force_red=True)
            for i in range(3, 5): add_ball(i, v2[0] + v3[0], v2[1] + v3[1], force_red=True)
            for i in range(5, 9): add_ball(i, v3[0], v3[1], force_red=True)
            for i in range(9, 14): add_ball(i, 0, 0, force_red=(frame_idx >= 70))
                
        rendered = draw_frame(current_state)
        rendered_rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        frames.append(rendered_rgb)

    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    main()
