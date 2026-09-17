import cv2
import numpy as np
import imageio
import os

def solve():
    first_frame = cv2.imread('/app/first_frame.png')
    cands = [
        (50, 829, 186, 186),
        (296, 829, 186, 186),
        (542, 829, 186, 186),
        (788, 829, 186, 186)
    ]
    
    pieces = []
    inpainted_bgs = []
    
    for cx, cy, cw, ch in cands:
        cand_img = first_frame[cy:cy+ch, cx:cx+cw]
        mask = ~np.all(cand_img >= [200, 200, 200], axis=-1)
        mask_dilate = cv2.dilate(mask.astype(np.uint8), np.ones((5,5), np.uint8), iterations=1)
        inpainted = cv2.inpaint(cand_img, mask_dilate, 3, cv2.INPAINT_TELEA)
        inpainted_bgs.append(inpainted)
        
        y_idx, x_idx = np.where(mask)
        if len(y_idx) > 0:
            min_y, max_y = y_idx.min(), y_idx.max()
            min_x, max_x = x_idx.min(), x_idx.max()
            piece_crop = cand_img[min_y:max_y+1, min_x:max_x+1]
            piece_mask = mask[min_y:max_y+1, min_x:max_x+1]
        else:
            min_x, min_y = 0, 0
            piece_crop = np.zeros((1,1,3), dtype=np.uint8)
            piece_mask = np.zeros((1,1), dtype=np.uint8)
            
        pieces.append((piece_crop, piece_mask, min_x, min_y))
        
    def draw_bg(img, cand_idx, status):
        cx, cy, cw, ch = cands[cand_idx]
        bg = inpainted_bgs[cand_idx].copy()
        
        if status == "yellow":
            cv2.rectangle(bg, (0, 0), (cw, ch), (0, 215, 255), 6)
        elif status == "red":
            overlay = bg.copy()
            overlay[:] = (0, 0, 255)
            bg = cv2.addWeighted(overlay, 0.2, bg, 0.8, 0)
            cv2.line(bg, (40, 40), (cw-40, ch-40), (0, 0, 255), 14)
            cv2.line(bg, (cw-40, 40), (40, ch-40), (0, 0, 255), 14)
            cv2.rectangle(bg, (0, 0), (cw, ch), (0, 0, 255), 6)
        elif status == "green":
            overlay = bg.copy()
            overlay[:] = (0, 255, 0)
            bg = cv2.addWeighted(overlay, 0.2, bg, 0.8, 0)
            cv2.line(bg, (40, ch//2), (cw//2-10, ch-40), (0, 255, 0), 14)
            cv2.line(bg, (cw//2-10, ch-40), (cw-30, 40), (0, 255, 0), 14)
            cv2.rectangle(bg, (0, 0), (cw, ch), (0, 255, 0), 6)
            
        img[cy:cy+ch, cx:cx+cw] = bg

    def paste_piece(img, piece_crop, piece_mask, x, y):
        h, w = piece_crop.shape[:2]
        y1, y2 = max(0, y), min(img.shape[0], y + h)
        x1, x2 = max(0, x), min(img.shape[1], x + w)
        
        if y1 >= y2 or x1 >= x2:
            return
            
        piece_y1 = y1 - y
        piece_y2 = piece_y1 + (y2 - y1)
        piece_x1 = x1 - x
        piece_x2 = piece_x1 + (x2 - x1)
        
        img_slice = img[y1:y2, x1:x2]
        crop_slice = piece_crop[piece_y1:piece_y2, piece_x1:piece_x2]
        mask_slice = piece_mask[piece_y1:piece_y2, piece_x1:piece_x2]
        
        img_slice[mask_slice > 0] = crop_slice[mask_slice > 0]

    frames = []
    hx, hy, hw, hh = 382, 331, 155, 155
    scale = 52.0 / 28.0
    
    for cand_idx in range(3):
        piece_crop, piece_mask, min_x, min_y = pieces[cand_idx]
        cx, cy, cw, ch = cands[cand_idx]
        
        new_w = int(round(piece_crop.shape[1] * scale))
        new_h = int(round(piece_crop.shape[0] * scale))
        resized_piece = cv2.resize(piece_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        resized_mask = cv2.resize(piece_mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        offset_x = hx + (hw - new_w) // 2
        offset_y = hy + (hh - new_h) // 2
        
        for frame_idx in range(16):
            frame = first_frame.copy()
            
            if frame_idx < 4:
                draw_bg(frame, cand_idx, "yellow")
                paste_piece(frame, piece_crop, piece_mask, cx + min_x, cy + min_y)
            elif frame_idx < 10:
                draw_bg(frame, cand_idx, "yellow")
                paste_piece(frame, piece_crop, piece_mask, cx + min_x, cy + min_y)
                paste_piece(frame, resized_piece, resized_mask, offset_x, offset_y)
            else:
                draw_bg(frame, cand_idx, "red")
                paste_piece(frame, piece_crop, piece_mask, cx + min_x, cy + min_y)
                paste_piece(frame, resized_piece, resized_mask, offset_x, offset_y)
                
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
    cand_idx = 3
    piece_crop, piece_mask, min_x, min_y = pieces[cand_idx]
    cx, cy, cw, ch = cands[cand_idx]
    
    new_w = int(round(piece_crop.shape[1] * scale))
    new_h = int(round(piece_crop.shape[0] * scale))
    resized_piece = cv2.resize(piece_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    resized_mask = cv2.resize(piece_mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    offset_x = hx + (hw - new_w) // 2
    offset_y = hy + (hh - new_h) // 2
    
    start_x = cx + min_x
    start_y = cy + min_y
    
    for frame_idx in range(16):
        frame = first_frame.copy()
        
        if frame_idx < 4:
            draw_bg(frame, cand_idx, "yellow")
            paste_piece(frame, piece_crop, piece_mask, cx + min_x, cy + min_y)
        elif frame_idx < 10:
            draw_bg(frame, cand_idx, "yellow")
            paste_piece(frame, piece_crop, piece_mask, cx + min_x, cy + min_y)
            paste_piece(frame, resized_piece, resized_mask, offset_x, offset_y)
        else:
            draw_bg(frame, cand_idx, "green")
            paste_piece(frame, piece_crop, piece_mask, cx + min_x, cy + min_y)
            paste_piece(frame, resized_piece, resized_mask, offset_x, offset_y)
            
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    for t in range(18):
        frame = first_frame.copy()
        draw_bg(frame, cand_idx, "green")
        
        f = t / 17.0
        cur_scale = 1.0 + f * (scale - 1.0)
        cur_w = int(round(piece_crop.shape[1] * cur_scale))
        cur_h = int(round(piece_crop.shape[0] * cur_scale))
        
        cur_x = int(round(start_x + f * (offset_x - start_x)))
        cur_y = int(round(start_y + f * (offset_y - start_y)))
        
        if cur_w == piece_crop.shape[1] and cur_h == piece_crop.shape[0]:
            frame_piece = piece_crop
            frame_mask = piece_mask
        else:
            frame_piece = cv2.resize(piece_crop, (cur_w, cur_h), interpolation=cv2.INTER_NEAREST)
            frame_mask = cv2.resize(piece_mask.astype(np.uint8), (cur_w, cur_h), interpolation=cv2.INTER_NEAREST)
            
        paste_piece(frame, frame_piece, frame_mask, cur_x, cur_y)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    for t in range(17):
        frame = first_frame.copy()
        draw_bg(frame, cand_idx, "green")
        paste_piece(frame, resized_piece, resized_mask, offset_x, offset_y)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', macro_block_size=None, pixelformat='yuv420p')
    for f in frames:
        writer.append_data(f)
    writer.close()

if __name__ == '__main__':
    solve()
