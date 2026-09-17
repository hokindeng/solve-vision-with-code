import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    
    # Define colors
    c_sq = np.array([250, 165, 96])
    c_diam = np.array([113, 113, 248])
    c_circ = np.array([238, 211, 34])
    c_out = np.array([139, 116, 100])
    c_bg = np.array([252, 250, 248])
    
    # Extract sprites
    def get_sprite_info(color):
        mask = cv2.inRange(img, color, color)
        y, x = np.where(mask > 0)
        x1, y1 = np.min(x), np.min(y)
        x2, y2 = np.max(x), np.max(y)
        sprite = img[y1:y2+1, x1:x2+1].copy()
        sprite_mask = mask[y1:y2+1, x1:x2+1].copy()
        return {
            'sprite': sprite,
            'mask': sprite_mask,
            'start_x': x1,
            'start_y': y1,
            'w': x2 - x1 + 1,
            'h': y2 - y1 + 1
        }

    sq_info = get_sprite_info(c_sq)
    diam_info = get_sprite_info(c_diam)
    circ_info = get_sprite_info(c_circ)
    
    # Find outlines
    mask_out = cv2.inRange(img, c_out, c_out)
    contours, _ = cv2.findContours(mask_out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outlines = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w / 2.0
        cy = y + h / 2.0
        outlines.append({'cx': cx, 'cy': cy})
    
    outlines.sort(key=lambda o: o['cy'])
    
    # Assign targets
    sq_info['target_x'] = int(round(outlines[0]['cx'] - sq_info['w'] / 2.0))
    sq_info['target_y'] = int(round(outlines[0]['cy'] - sq_info['h'] / 2.0))
    
    diam_info['target_x'] = int(round(outlines[1]['cx'] - diam_info['w'] / 2.0))
    diam_info['target_y'] = int(round(outlines[1]['cy'] - diam_info['h'] / 2.0))
    
    circ_info['target_x'] = int(round(outlines[2]['cx'] - circ_info['w'] / 2.0))
    circ_info['target_y'] = int(round(outlines[2]['cy'] - circ_info['h'] / 2.0))
    
    # Create empty background
    bg = img.copy()
    for c in [c_sq, c_diam, c_circ]:
        mask = cv2.inRange(img, c, c)
        bg[mask > 0] = c_bg
        
    # Animation parameters
    # Total frames ~78
    # 3 moves, we can do 15 frames per move, and some pauses.
    # Let's say:
    # 0..4: wait (5 frames)
    # 5..19: move sq (15 frames)
    # 20..24: wait (5 frames)
    # 25..39: move diam (15 frames)
    # 40..44: wait (5 frames)
    # 45..59: move circ (15 frames)
    # 60..77: wait (18 frames) -> total 78 frames
    
    def blend(start, end, t):
        return int(round(start + (end - start) * t))
    
    frames = []
    
    # We maintain current positions of each shape
    curr_sq = (sq_info['start_x'], sq_info['start_y'])
    curr_diam = (diam_info['start_x'], diam_info['start_y'])
    curr_circ = (circ_info['start_x'], circ_info['start_y'])
    
    for f in range(78):
        # determine positions
        if f < 5:
            pass # wait
        elif f < 20:
            t = (f - 4) / 15.0 # 1 to 15 / 15
            curr_sq = (blend(sq_info['start_x'], sq_info['target_x'], t),
                       blend(sq_info['start_y'], sq_info['target_y'], t))
        elif f < 25:
            curr_sq = (sq_info['target_x'], sq_info['target_y'])
        elif f < 40:
            t = (f - 24) / 15.0
            curr_diam = (blend(diam_info['start_x'], diam_info['target_x'], t),
                         blend(diam_info['start_y'], diam_info['target_y'], t))
        elif f < 45:
            curr_diam = (diam_info['target_x'], diam_info['target_y'])
        elif f < 60:
            t = (f - 44) / 15.0
            curr_circ = (blend(circ_info['start_x'], circ_info['target_x'], t),
                         blend(circ_info['start_y'], circ_info['target_y'], t))
        else:
            curr_circ = (circ_info['target_x'], circ_info['target_y'])
            
        # draw frame
        frame = bg.copy()
        
        # draw shapes in order: circ, diam, sq (moving one drawn last if overlapping, but they don't overlap)
        # Actually it's better to draw them in a fixed order.
        for info, pos in [(circ_info, curr_circ), (diam_info, curr_diam), (sq_info, curr_sq)]:
            x, y = pos
            sp = info['sprite']
            sp_mask = info['mask']
            # safely paste (assuming no out of bounds, since coordinates are within image)
            frame[y:y+info['h'], x:x+info['w']][sp_mask > 0] = sp[sp_mask > 0]
            
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    os.makedirs('/app/output', exist_ok=True)
    # Write video
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, macro_block_size=None, codec='libx264', pixelformat='yuv420p')
    print("Video generated successfully.")

if __name__ == '__main__':
    solve()
