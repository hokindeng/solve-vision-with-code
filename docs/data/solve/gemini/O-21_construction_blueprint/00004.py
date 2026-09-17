import cv2
import numpy as np
import imageio

def get_piece(img, box):
    x0, y0, w, h = box
    piece_img = img[y0:y0+h, x0:x0+w].copy()
    mask = np.zeros((h, w), dtype=bool)
    for yy in range(h):
        for xx in range(w):
            b, g, r = piece_img[yy, xx]
            if not ((b==230 and g==230 and r==230) or (b==245 and g==245 and r==245)):
                mask[yy, xx] = True
    return piece_img, mask

def draw_piece(img, piece_img, mask, cx, cy):
    h, w = mask.shape
    tx = cx - w // 2
    ty = cy - h // 2
    for yy in range(h):
        for xx in range(w):
            if mask[yy, xx]:
                if 0 <= ty+yy < img.shape[0] and 0 <= tx+xx < img.shape[1]:
                    img[ty+yy, tx+xx] = piece_img[yy, xx]

def erase_piece(img, box, mask):
    x0, y0, w, h = box
    for yy in range(h):
        for xx in range(w):
            if mask[yy, xx]:
                img[y0+yy, x0+xx] = (230, 230, 230)

def draw_cross(img, box):
    x0, y0, w, h = box
    cx, cy = x0 + w // 2, y0 + h // 2
    s = 40
    cv2.line(img, (cx-s, cy-s), (cx+s, cy+s), (0, 0, 255), 8)
    cv2.line(img, (cx-s, cy+s), (cx+s, cy-s), (0, 0, 255), 8)

def draw_check(img, box):
    x0, y0, w, h = box
    cx, cy = x0 + w // 2, y0 + h // 2
    cv2.line(img, (cx-20, cy), (cx-5, cy+20), (0, 255, 0), 8)
    cv2.line(img, (cx-5, cy+20), (cx+30, cy-20), (0, 255, 0), 8)

def main():
    first_frame = cv2.imread('/app/first_frame.png')
    
    # Pre-compute piece masks
    boxes = [(50, 829, 186, 186), (296, 829, 186, 186), (542, 829, 186, 186), (788, 829, 186, 186)]
    pieces = [get_piece(first_frame, box) for box in boxes]
    
    gap_cx, gap_cy = 407, 434
    correct_idx = 2
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    # state frame counter
    fc = 0
    
    for i in range(4):
        box = boxes[i]
        piece_img, piece_mask = pieces[i]
        
        # Highlight (6 frames)
        for _ in range(6):
            frame = first_frame.copy()
            # Restore previous judgments if any
            for j in range(i):
                bx = boxes[j]
                cv2.rectangle(frame, (bx[0], bx[1]), (bx[0]+bx[2], bx[1]+bx[3]), (0, 0, 255), 6)
                draw_cross(frame, bx)
                
            cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 255), 6)
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            fc += 1
            
        # Preview (9 frames)
        for _ in range(9):
            frame = first_frame.copy()
            for j in range(i):
                bx = boxes[j]
                cv2.rectangle(frame, (bx[0], bx[1]), (bx[0]+bx[2], bx[1]+bx[3]), (0, 0, 255), 6)
                draw_cross(frame, bx)
            cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 255), 6)
            draw_piece(frame, piece_img, piece_mask, gap_cx, gap_cy)
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            fc += 1
            
        # Judge
        if i == correct_idx:
            # Judge Green (9 frames)
            for _ in range(9):
                frame = first_frame.copy()
                for j in range(i):
                    bx = boxes[j]
                    cv2.rectangle(frame, (bx[0], bx[1]), (bx[0]+bx[2], bx[1]+bx[3]), (0, 0, 255), 6)
                    draw_cross(frame, bx)
                cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 6)
                draw_check(frame, box)
                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                fc += 1
                
            # Animation (18 frames)
            start_cx, start_cy = box[0] + box[2]//2, box[1] + box[3]//2
            for t_step in range(1, 19):
                t = t_step / 18.0
                cx = int(start_cx + (gap_cx - start_cx) * t)
                cy = int(start_cy + (gap_cy - start_cy) * t)
                
                frame = first_frame.copy()
                # Draw previous judgments
                for j in range(i):
                    bx = boxes[j]
                    cv2.rectangle(frame, (bx[0], bx[1]), (bx[0]+bx[2], bx[1]+bx[3]), (0, 0, 255), 6)
                    draw_cross(frame, bx)
                
                # Erase piece from its box
                erase_piece(frame, box, piece_mask)
                
                # Keep green box and check
                cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 6)
                draw_check(frame, box)
                
                # draw moving piece
                draw_piece(frame, piece_img, piece_mask, cx, cy)
                
                # if near end, optionally remove red dash? Let's just do it at the very end
                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                fc += 1
                
            # Hold (9 frames)
            for _ in range(9):
                frame = first_frame.copy()
                for j in range(i):
                    bx = boxes[j]
                    cv2.rectangle(frame, (bx[0], bx[1]), (bx[0]+bx[2], bx[1]+bx[3]), (0, 0, 255), 6)
                    draw_cross(frame, bx)
                erase_piece(frame, box, piece_mask)
                cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 6)
                draw_check(frame, box)
                
                # Remove red dashes
                red_mask = ((frame[:,:,0] == 100) & (frame[:,:,1] == 100) & (frame[:,:,2] == 255))
                frame[red_mask] = (245, 245, 245)
                
                # Draw piece in gap
                draw_piece(frame, piece_img, piece_mask, gap_cx, gap_cy)
                
                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                fc += 1
            
            break # Stop after finding the matching piece!
            
        else:
            # Judge Red (9 frames)
            for _ in range(9):
                frame = first_frame.copy()
                for j in range(i+1):
                    bx = boxes[j]
                    cv2.rectangle(frame, (bx[0], bx[1]), (bx[0]+bx[2], bx[1]+bx[3]), (0, 0, 255), 6)
                    draw_cross(frame, bx)
                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                fc += 1

    writer.close()
    print(f"Total frames written: {fc}")

if __name__ == '__main__':
    main()
