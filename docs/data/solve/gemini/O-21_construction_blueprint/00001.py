import cv2
import numpy as np
import imageio

original_img = cv2.imread('/app/first_frame.png')

def erase_r_line(img_target):
    r_mask = cv2.inRange(img_target, np.array([100, 100, 255]), np.array([100, 100, 255]))
    img_target[r_mask > 0] = [245, 245, 245]

def highlight_cand(img_target, i):
    cx = [51, 297, 543, 789][i]
    cv2.rectangle(img_target, (cx, 830), (cx+184, 1014), (0, 215, 255), 6)

def draw_judge(img_target, i, match):
    cx = [51, 297, 543, 789][i]
    color = (0, 200, 0) if match else (0, 0, 255)
    cv2.rectangle(img_target, (cx, 830), (cx+184, 1014), color, 6)
    
    mark_center = (cx + 150, 980)
    if match:
        p1 = (mark_center[0] - 15, mark_center[1] - 5)
        p2 = (mark_center[0] - 5, mark_center[1] + 15)
        p3 = (mark_center[0] + 15, mark_center[1] - 15)
        cv2.line(img_target, p1, p2, color, 6, cv2.LINE_AA)
        cv2.line(img_target, p2, p3, color, 6, cv2.LINE_AA)
    else:
        cv2.line(img_target, (mark_center[0]-15, mark_center[1]-15), (mark_center[0]+15, mark_center[1]+15), color, 6, cv2.LINE_AA)
        cv2.line(img_target, (mark_center[0]-15, mark_center[1]+15), (mark_center[0]+15, mark_center[1]-15), color, 6, cv2.LINE_AA)

def get_piece_info(i):
    cx_box = [51, 297, 543, 789][i]
    cand = original_img[830:1014, cx_box:cx_box+184]
    mask_bg = cv2.inRange(cand, np.array([244, 244, 244]), np.array([246, 246, 246]))
    mask_bd = cv2.inRange(cand, np.array([229, 229, 231]), np.array([231, 231, 231]))
    mask = cv2.bitwise_not(cv2.bitwise_or(mask_bg, mask_bd))
    
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0: return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    piece = cand[y_min:y_max+1, x_min:x_max+1]
    piece_mask = mask[y_min:y_max+1, x_min:x_max+1]
    
    return piece, piece_mask, y_min, y_max, x_min, x_max

def preview_cand(i, img_target):
    info = get_piece_info(i)
    if not info: return
    piece, piece_mask = info[:2]
    
    scale = 1.405
    piece_scaled = cv2.resize(piece, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    mask_scaled = cv2.resize(piece_mask, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    
    th, tw = piece_scaled.shape[:2]
    ty = 305 - th // 2
    tx = 434 - tw // 2
    
    # We shouldn't permanently erase R line for translucent preview, 
    # but the prompt says "previews how the piece fits".
    # I'll alpha blend the piece over the gap
    alpha = 0.6
    
    for r in range(th):
        for c in range(tw):
            if mask_scaled[r, c] > 0:
                if 0 <= ty+r < img_target.shape[0] and 0 <= tx+c < img_target.shape[1]:
                    bg = img_target[ty+r, tx+c]
                    fg = piece_scaled[r, c]
                    img_target[ty+r, tx+c] = (fg * alpha + bg * (1 - alpha)).astype(np.uint8)

def erase_cand(img_target, i):
    cx = [51, 297, 543, 789][i]
    img_target[832:1012, cx+2:cx+182] = [245, 245, 245]

def draw_moving_cand(img_target, progress):
    info = get_piece_info(3)
    if not info: return
    piece, piece_mask, y_min, y_max, x_min, x_max = info
    
    scale = 1.0 + (1.405 - 1.0) * progress
    piece_scaled = cv2.resize(piece, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    mask_scaled = cv2.resize(piece_mask, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    
    th, tw = piece_scaled.shape[:2]
    
    cy_orig = 830 + (y_min + y_max) // 2
    cx_orig = 789 + (x_min + x_max) // 2
    
    cy_target = 305
    cx_target = 434
    
    cy_curr = cy_orig + (cy_target - cy_orig) * progress
    cx_curr = cx_orig + (cx_target - cx_orig) * progress
    
    ty = int(cy_curr - th / 2)
    tx = int(cx_curr - tw / 2)
    
    erase_r_line(img_target)
    
    for r in range(th):
        for c in range(tw):
            if mask_scaled[r, c] > 0:
                if 0 <= ty+r < img_target.shape[0] and 0 <= tx+c < img_target.shape[1]:
                    img_target[ty+r, tx+c] = piece_scaled[r, c]

def draw_perfect_blocks(img_target):
    erase_r_line(img_target)
    blocks = [
        (434, 227),
        (382, 279),
        (434, 279),
        (434, 331)
    ]
    for x, y in blocks:
        cv2.rectangle(img_target, (x, y), (x+50, y+50), (50, 100, 140), -1) 
        cv2.rectangle(img_target, (x+1, y+1), (x+49, y+49), (200, 100, 0), -1) 

frames = []

for f in range(99):
    res = original_img.copy()
    
    for prev_i, prev_f, match in [(0, 15, False), (1, 30, False), (2, 45, False), (3, 60, True)]:
        if f >= prev_f:
            draw_judge(res, prev_i, match)
            if prev_i == 3:
                erase_cand(res, 3)
    
    if f == 0:
        pass
    elif f < 15:
        if f < 5:
            highlight_cand(res, 0)
        elif 5 <= f < 10:
            highlight_cand(res, 0)
            preview_cand(0, res)
        else:
            draw_judge(res, 0, False)
            preview_cand(0, res)
    elif f < 30:
        if f < 20:
            highlight_cand(res, 1)
        elif 20 <= f < 25:
            highlight_cand(res, 1)
            preview_cand(1, res)
        else:
            draw_judge(res, 1, False)
            preview_cand(1, res)
    elif f < 45:
        if f < 35:
            highlight_cand(res, 2)
        elif 35 <= f < 40:
            highlight_cand(res, 2)
            preview_cand(2, res)
        else:
            draw_judge(res, 2, False)
            preview_cand(2, res)
    else:
        if f < 50:
            highlight_cand(res, 3)
        elif 50 <= f < 55:
            highlight_cand(res, 3)
            preview_cand(3, res)
        elif 55 <= f < 60:
            draw_judge(res, 3, True)
            preview_cand(3, res)
        elif 60 <= f < 91:
            progress = (f - 60) / 30.0
            draw_moving_cand(res, progress)
        else:
            draw_perfect_blocks(res)
            
    res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    frames.append(res_rgb)

imageio.mimwrite('/app/output/video.mp4', frames, fps=16, macro_block_size=None, format='FFMPEG', codec='libx264', pixelformat='yuv420p')
