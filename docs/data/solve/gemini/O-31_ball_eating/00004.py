import cv2
import numpy as np
import imageio
import os

def main():
    # Load original image
    img = cv2.imread('/app/first_frame.png')
    
    bg_color = img[0,0]
    diff = np.any(img != bg_color, axis=2).astype(np.uint8) * 255
    
    # Use OpenCV for connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(diff, connectivity=8)

    balls = {}
    for i in range(1, num_labels):
        mask = (labels == i)
        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = centroids[i]
        balls[i] = {
            'mask': mask,
            'cx': cx,
            'cy': cy,
            'area': area
        }

    # Known sizes and locations derived from the first frame
    known_balls = {
        'B2': {'cx': 583.0, 'cy': 313.0, 'R': 28.5},
        'black': {'cx': 651.0, 'cy': 371.0, 'R': 41.5},
        'B3': {'cx': 153.0, 'cy': 382.0, 'R': 60.5},
        'B6': {'cx': 120.0, 'cy': 918.0, 'R': 84.5},
        'B5': {'cx': 754.0, 'cy': 540.0, 'R': 99.5},
        'B1': {'cx': 830.0, 'cy': 142.0, 'R': 99.5}
    }

    def get_mask_for(cx, cy):
        min_d = 1e9
        best_mask = None
        for i, b in balls.items():
            d = (b['cx']-cx)**2 + (b['cy']-cy)**2
            if d < min_d:
                min_d = d
                best_mask = b['mask']
        return best_mask

    masks = {}
    for k, v in known_balls.items():
        masks[k] = get_mask_for(v['cx'], v['cy'])

    # Prepare background with black ball removed
    bg_img = img.copy()
    bg_img[masks['black']] = [255, 255, 255]

    sequence = ['B2', 'B3', 'B6', 'B5', 'B1']
    move_frames_list = [4, 18, 22, 30, 17]
    grow_frames = 3

    frames = []
    # Frame 0
    frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    curr_cx = known_balls['black']['cx']
    curr_cy = known_balls['black']['cy']
    curr_R = known_balls['black']['R']

    for step_idx, target_name in enumerate(sequence):
        target = known_balls[target_name]
        move_f = move_frames_list[step_idx]
        
        start_cx, start_cy = curr_cx, curr_cy
        end_cx, end_cy = target['cx'], target['cy']
        
        # Move phase
        for f in range(1, move_f + 1):
            alpha = f / move_f
            cx = start_cx + (end_cx - start_cx) * alpha
            cy = start_cy + (end_cy - start_cy) * alpha
            
            if f == move_f:
                bg_img[masks[target_name]] = [255, 255, 255]
            
            frame = bg_img.copy()
            cv2.circle(frame, (int(cx * 16), int(cy * 16)), int(curr_R * 16), (0, 0, 0), -1, cv2.LINE_AA, shift=4)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
        # Grow phase
        start_R = curr_R
        end_R = curr_R + target['R']
        for f in range(1, grow_frames + 1):
            alpha = f / grow_frames
            curr_R = start_R + (end_R - start_R) * alpha
            
            frame = bg_img.copy()
            cv2.circle(frame, (int(end_cx * 16), int(end_cy * 16)), int(curr_R * 16), (0, 0, 0), -1, cv2.LINE_AA, shift=4)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
        curr_cx, curr_cy = end_cx, end_cy
        curr_R = end_R

    # Hold frame
    frame = bg_img.copy()
    cv2.circle(frame, (int(curr_cx * 16), int(curr_cy * 16)), int(curr_R * 16), (0, 0, 0), -1, cv2.LINE_AA, shift=4)
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, macro_block_size=None, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    main()
