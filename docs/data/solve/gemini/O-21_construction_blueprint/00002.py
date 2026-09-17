import cv2
import numpy as np
import imageio

first_frame = cv2.imread('/app/first_frame.png')
def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

frames = []

cand_bboxes = [
    (85, 883, 113, 75),
    (350, 902, 75, 37),
    (596, 883, 75, 75),
    (842, 883, 75, 75)
]
target_sizes = [(155, 103), (103, 51), (103, 103), (103, 103)]
boxes = [(51, 830, 184, 184), (297, 830, 184, 184), (543, 830, 184, 184), (789, 830, 184, 184)]

def get_cand_unscaled(idx):
    x, y, w, h = cand_bboxes[idx]
    crop = first_frame[y:y+h, x:x+w]
    mask = (crop == [230,230,230]).all(axis=2) | (crop == [245,245,245]).all(axis=2)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = crop
    rgba[..., 3] = np.where(mask, 0, 255)
    return rgba

def get_cand(idx):
    rgba = get_cand_unscaled(idx)
    return cv2.resize(rgba, target_sizes[idx], interpolation=cv2.INTER_NEAREST)

def overlay(bg, fg, x, y, alpha=0.8):
    fh, fw = fg.shape[:2]
    roi = bg[y:y+fh, x:x+fw]
    fg_a = (fg[..., 3]/255.0)*alpha
    bg_a = 1.0 - fg_a
    for c in range(3):
        roi[..., c] = (roi[..., c] * bg_a + fg[..., c] * fg_a).astype(np.uint8)

def draw_box(frame, idx, state):
    x, y, w, h = boxes[idx]
    cx, cy = x + w//2, y + h//2
    
    if state == "highlight":
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 200, 50), 6) # Light blue in BGR
    elif state == "wrong":
        inner = frame[y:y+h, x:x+w]
        mask = (inner == [230, 230, 230]).all(axis=2)
        inner[mask] = [200, 200, 255] # Light red BGR
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 6)
        cv2.line(frame, (cx-40, cy-40), (cx+40, cy+40), (0, 0, 255), 10, lineType=cv2.LINE_AA)
        cv2.line(frame, (cx+40, cy-40), (cx-40, cy+40), (0, 0, 255), 10, lineType=cv2.LINE_AA)
    elif state == "right":
        inner = frame[y:y+h, x:x+w]
        mask = (inner == [230, 230, 230]).all(axis=2)
        inner[mask] = [200, 255, 200]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 6)
        cv2.line(frame, (cx-30, cy), (cx-10, cy+20), (0, 255, 0), 10, lineType=cv2.LINE_AA)
        cv2.line(frame, (cx-10, cy+20), (cx+40, cy-30), (0, 255, 0), 10, lineType=cv2.LINE_AA)
    elif state == "right_empty":
        cv2.rectangle(frame, (x, y), (x+w, y+h), (200, 255, 200), -1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 6)
        cv2.line(frame, (cx-30, cy), (cx-10, cy+20), (0, 255, 0), 10, lineType=cv2.LINE_AA)
        cv2.line(frame, (cx-10, cy+20), (cx+40, cy-30), (0, 255, 0), 10, lineType=cv2.LINE_AA)

for i in range(4):
    if i < 3:
        for f in range(20):
            frame = first_frame.copy()
            if i == 0 and f == 0:
                pass # First frame untouched
            else:
                for j in range(4):
                    if j < i:
                        draw_box(frame, j, "wrong")
                
                if 0 <= f < 5:
                    draw_box(frame, i, "highlight")
                elif 5 <= f < 14:
                    draw_box(frame, i, "highlight")
                    overlay(frame, get_cand(i), 486, 383, alpha=0.8)
                elif 14 <= f < 20:
                    draw_box(frame, i, "wrong")
            frames.append(bgr2rgb(frame))
    else:
        for f in range(39):
            frame = first_frame.copy()
            for j in range(3):
                draw_box(frame, j, "wrong")
            
            if 0 <= f < 5:
                draw_box(frame, 3, "highlight")
            elif 5 <= f < 14:
                draw_box(frame, 3, "highlight")
                overlay(frame, get_cand(3), 486, 383, alpha=0.8)
            elif 14 <= f < 20:
                draw_box(frame, 3, "right")
            elif 20 <= f < 38:
                draw_box(frame, 3, "right_empty")
                
                mask = cv2.inRange(frame, np.array([100, 100, 255]), np.array([100, 100, 255]))
                frame[mask > 0] = [245, 245, 245]
                
                progress = (f - 20) / 17.0
                
                start_x, start_y = cand_bboxes[3][0], cand_bboxes[3][1]
                end_x, end_y = 486, 383
                
                curr_x = int(start_x * (1 - progress) + end_x * progress)
                curr_y = int(start_y * (1 - progress) + end_y * progress)
                
                start_w, start_h = cand_bboxes[3][2], cand_bboxes[3][3]
                end_w, end_h = target_sizes[3][0], target_sizes[3][1]
                
                curr_w = int(start_w * (1 - progress) + end_w * progress)
                curr_h = int(start_h * (1 - progress) + end_h * progress)
                
                orig_cand = get_cand_unscaled(3)
                curr_cand = cv2.resize(orig_cand, (curr_w, curr_h), interpolation=cv2.INTER_NEAREST)
                
                overlay(frame, curr_cand, curr_x, curr_y, alpha=1.0)
            elif f == 38:
                draw_box(frame, 3, "right_empty")
                
                mask = cv2.inRange(frame, np.array([100, 100, 255]), np.array([100, 100, 255]))
                frame[mask > 0] = [245, 245, 245]
                
                overlay(frame, get_cand(3), 486, 383, alpha=1.0)
                
            frames.append(bgr2rgb(frame))

writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', macro_block_size=None, pixelformat='yuv420p')
for frm in frames:
    writer.append_data(frm)
writer.close()
