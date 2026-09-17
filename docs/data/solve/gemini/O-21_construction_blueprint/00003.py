import cv2
import numpy as np
import imageio
import os

def draw_piece(out, p_img, p_mask, px, py):
    sh, sw = p_img.shape[:2]
    # Handle out of bounds
    out_h, out_w = out.shape[:2]
    y1 = max(0, py)
    y2 = min(out_h, py + sh)
    x1 = max(0, px)
    x2 = min(out_w, px + sw)
    
    if y1 >= y2 or x1 >= x2:
        return
        
    p_y1 = y1 - py
    p_y2 = p_y1 + (y2 - y1)
    p_x1 = x1 - px
    p_x2 = p_x1 + (x2 - x1)
    
    sub_mask = p_mask[p_y1:p_y2, p_x1:p_x2]
    sub_img = p_img[p_y1:p_y2, p_x1:p_x2]
    sub_out = out[y1:y2, x1:x2]
    
    for c in range(3):
        sub_out[:,:,c] = np.where(sub_mask, sub_img[:,:,c], sub_out[:,:,c])

def solve():
    img = cv2.imread('/app/first_frame.png')
    
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    
    gap_cx, gap_cy = 486 + 52, 383 + 26
    clean_block = img[383:434+1, 382:485+1].copy()
    
    for f in range(99):
        out = img.copy()
        
        # Determine current candidate based on frame
        if f < 18:
            cand_idx = 0
            local_f = f
        elif f < 36:
            cand_idx = 1
            local_f = f - 18
        elif f < 54:
            cand_idx = 2
            local_f = f - 36
        else:
            cand_idx = 3
            local_f = f - 54
            
        bx = 50 + 246 * cand_idx
        
        # extract candidate piece
        roi = img[829+50:829+186-10, bx+10:bx+186-10]
        mask = (roi[:,:,0] != 230) | (roi[:,:,1] != 230) | (roi[:,:,2] != 230)
        ys, xs = np.where(mask)
        c_y1, c_y2 = ys.min(), ys.max()
        c_x1, c_x2 = xs.min(), xs.max()
        piece = roi[c_y1:c_y2+1, c_x1:c_x2+1].copy()
        piece_mask = mask[c_y1:c_y2+1, c_x1:c_x2+1].copy()
        
        piece_orig_x = bx + 10 + c_x1
        piece_orig_y = 829 + 50 + c_y1
        piece_cx = piece_orig_x + piece.shape[1] // 2
        piece_cy = piece_orig_y + piece.shape[0] // 2
        
        scale = 52 / 37.0
        
        if cand_idx < 3:
            if local_f < 4:
                cv2.rectangle(out, (bx+2, 829+2), (bx+186-2, 829+186-2), (0, 255, 255), 6)
            elif local_f < 12:
                cv2.rectangle(out, (bx+2, 829+2), (bx+186-2, 829+186-2), (0, 255, 255), 6)
                sw, sh = int(piece.shape[1] * scale), int(piece.shape[0] * scale)
                p_img = cv2.resize(piece, (sw, sh), interpolation=cv2.INTER_NEAREST)
                p_mask = cv2.resize(piece_mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
                px, py = gap_cx - sw // 2, gap_cy - sh // 2
                draw_piece(out, p_img, p_mask, px, py)
            else:
                cv2.rectangle(out, (bx+2, 829+2), (bx+186-2, 829+186-2), (0, 0, 220), 6)
                x, y = bx + 186 - 35, 829 + 35
                cv2.line(out, (x-15, y-15), (x+15, y+15), (0, 0, 220), 8)
                cv2.line(out, (x+15, y-15), (x-15, y+15), (0, 0, 220), 8)
                sw, sh = int(piece.shape[1] * scale), int(piece.shape[0] * scale)
                p_img = cv2.resize(piece, (sw, sh), interpolation=cv2.INTER_NEAREST)
                p_mask = cv2.resize(piece_mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
                px, py = gap_cx - sw // 2, gap_cy - sh // 2
                draw_piece(out, p_img, p_mask, px, py)
        else:
            if local_f >= 18:
                # erase original piece
                sub_out = out[piece_orig_y:piece_orig_y+piece.shape[0], piece_orig_x:piece_orig_x+piece.shape[1]]
                for c in range(3):
                    sub_out[:,:,c] = np.where(piece_mask, 230, sub_out[:,:,c])
                    
            if local_f < 4:
                cv2.rectangle(out, (bx+2, 829+2), (bx+186-2, 829+186-2), (0, 255, 255), 6)
            elif local_f < 12:
                cv2.rectangle(out, (bx+2, 829+2), (bx+186-2, 829+186-2), (0, 255, 255), 6)
                sw, sh = int(piece.shape[1] * scale), int(piece.shape[0] * scale)
                p_img = cv2.resize(piece, (sw, sh), interpolation=cv2.INTER_NEAREST)
                p_mask = cv2.resize(piece_mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
                px, py = gap_cx - sw // 2, gap_cy - sh // 2
                draw_piece(out, p_img, p_mask, px, py)
            elif local_f < 18:
                cv2.rectangle(out, (bx+2, 829+2), (bx+186-2, 829+186-2), (0, 200, 0), 6)
                x, y = bx + 186 - 35, 829 + 35
                cv2.line(out, (x-10, y), (x+5, y+15), (0, 200, 0), 8)
                cv2.line(out, (x+5, y+15), (x+25, y-15), (0, 200, 0), 8)
                sw, sh = int(piece.shape[1] * scale), int(piece.shape[0] * scale)
                p_img = cv2.resize(piece, (sw, sh), interpolation=cv2.INTER_NEAREST)
                p_mask = cv2.resize(piece_mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
                px, py = gap_cx - sw // 2, gap_cy - sh // 2
                draw_piece(out, p_img, p_mask, px, py)
            elif local_f < 38:
                cv2.rectangle(out, (bx+2, 829+2), (bx+186-2, 829+186-2), (0, 200, 0), 6)
                x, y = bx + 186 - 35, 829 + 35
                cv2.line(out, (x-10, y), (x+5, y+15), (0, 200, 0), 8)
                cv2.line(out, (x+5, y+15), (x+25, y-15), (0, 200, 0), 8)
                
                t = (local_f - 18) / 19.0
                t = max(0, min(1, t))
                t = t * t * (3 - 2 * t)
                
                curr_scale = 1.0 + (scale - 1.0) * t
                curr_cx = int(piece_cx + (gap_cx - piece_cx) * t)
                curr_cy = int(piece_cy + (gap_cy - piece_cy) * t)
                
                sw, sh = int(piece.shape[1] * curr_scale), int(piece.shape[0] * curr_scale)
                if sw > 0 and sh > 0:
                    p_img = cv2.resize(piece, (sw, sh), interpolation=cv2.INTER_NEAREST)
                    p_mask = cv2.resize(piece_mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
                    px, py = curr_cx - sw // 2, curr_cy - sh // 2
                    draw_piece(out, p_img, p_mask, px, py)
            else:
                cv2.rectangle(out, (bx+2, 829+2), (bx+186-2, 829+186-2), (0, 200, 0), 6)
                x, y = bx + 186 - 35, 829 + 35
                cv2.line(out, (x-10, y), (x+5, y+15), (0, 200, 0), 8)
                cv2.line(out, (x+5, y+15), (x+25, y-15), (0, 200, 0), 8)
                out[383:434+1, 486:589+1] = clean_block
                
        # convert BGR to RGB for imageio
        out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        writer.append_data(out_rgb)
        
    writer.close()

if __name__ == '__main__':
    solve()
