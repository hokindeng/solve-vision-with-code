import cv2
import numpy as np
import imageio
import os

def get_topmost_polygon(img):
    colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)
    
    counts = {}
    for c in colors:
        counts[tuple(c)] = np.count_nonzero(np.all(img == c, axis=-1))
    
    bg_color = max(counts, key=counts.get)
    
    topmost_color = None
    max_solidity = -1
    best_approx = None
    
    for c in colors:
        c_tuple = tuple(c)
        if c_tuple == bg_color: continue
        
        mask = np.all(img == c, axis=-1).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        cnt = max(contours, key=cv2.contourArea)
        
        peri = cv2.arcLength(cnt, True)
        
        for eps in [0.01, 0.02, 0.03]:
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            approx_mask = np.zeros_like(mask)
            cv2.drawContours(approx_mask, [approx], -1, 255, -1)
            
            mask_area = np.count_nonzero(mask)
            approx_area = np.count_nonzero(approx_mask)
            if max(mask_area, approx_area) == 0: continue
            
            ratio = min(mask_area, approx_area) / max(mask_area, approx_area)
            if ratio > max_solidity:
                max_solidity = ratio
                topmost_color = c
                best_approx = approx
                
    return best_approx

def generate_video():
    first_frame_path = '/app/first_frame.png'
    output_path = '/app/output/video.mp4'
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    img = cv2.imread(first_frame_path)
    if img is None:
        raise FileNotFoundError(f"Could not read {first_frame_path}")
        
    approx = get_topmost_polygon(img)
    pts = [tuple(v[0]) for v in approx]
    
    num_frames = 40
    fps = 16
    
    segments = []
    total_len = 0
    for i in range(len(pts)):
        p1 = pts[i]
        p2 = pts[(i+1)%len(pts)]
        length = np.linalg.norm(np.array(p1) - np.array(p2))
        segments.append({'p1': p1, 'p2': p2, 'len': length})
        total_len += length
        
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    for i in range(num_frames):
        frame = img.copy()
        
        progress = i / (num_frames - 1)
        target_len = progress * total_len
        
        current_len = 0
        for seg in segments:
            if current_len >= target_len:
                break
                
            p1 = seg['p1']
            p2 = seg['p2']
            seg_len = seg['len']
            
            if current_len + seg_len <= target_len:
                cv2.line(frame, p1, p2, (0, 0, 255), 5, lineType=cv2.LINE_AA)
                current_len += seg_len
            else:
                remaining = target_len - current_len
                fraction = remaining / seg_len
                p_end = (
                    int(round(p1[0] + (p2[0] - p1[0]) * fraction)),
                    int(round(p1[1] + (p2[1] - p1[1]) * fraction))
                )
                cv2.line(frame, p1, p_end, (0, 0, 255), 5, lineType=cv2.LINE_AA)
                break
                
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    generate_video()
