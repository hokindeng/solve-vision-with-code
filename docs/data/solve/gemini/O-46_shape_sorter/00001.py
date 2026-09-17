import cv2
import numpy as np
import imageio
import math
import os

def main():
    img = cv2.imread('/app/first_frame.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    bg_color = np.array([248, 250, 252])
    
    diff = np.abs(img.astype(np.float32) - bg_color)
    mask = (np.sum(diff, axis=2) > 0).astype(np.uint8) * 255
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    cards_info = []
    outlines_info = []
    for i in range(2, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 3000:
            cards_info.append(i)
        else:
            outlines_info.append(i)
            
    assert len(cards_info) == 3
    assert len(outlines_info) == 3
    
    orange_square = None
    pink_diamond = None
    yellow_circle = None
    for i in cards_info:
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        if y > 500:
            yellow_circle = i
        elif x < 200:
            orange_square = i
        else:
            pink_diamond = i
            
    square_outline = None
    diamond_outline = None
    circle_outline = None
    for i in outlines_info:
        y = stats[i, cv2.CC_STAT_TOP]
        if y < 350:
            square_outline = i
        elif y < 550:
            diamond_outline = i
        else:
            circle_outline = i
            
    def get_center(comp_id):
        x = stats[comp_id, cv2.CC_STAT_LEFT]
        y = stats[comp_id, cv2.CC_STAT_TOP]
        w = stats[comp_id, cv2.CC_STAT_WIDTH]
        h = stats[comp_id, cv2.CC_STAT_HEIGHT]
        return x + w / 2.0, y + h / 2.0
        
    targets = {}
    for card, outline in [(orange_square, square_outline),
                          (pink_diamond, diamond_outline),
                          (yellow_circle, circle_outline)]:
        cx_c, cy_c = get_center(card)
        cx_o, cy_o = get_center(outline)
        targets[card] = (cx_o - cx_c, cy_o - cy_c)
        
    bg_canvas = img.copy()
    for comp in [orange_square, pink_diamond, yellow_circle]:
        bg_canvas[labels == comp] = bg_color
        
    cards = {}
    for comp in [orange_square, pink_diamond, yellow_circle]:
        card_rgba = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.float32)
        card_rgba[:, :, :3] = img.astype(np.float32)
        card_mask = (labels == comp)
        alpha = card_mask.astype(np.float32)
        card_rgba[:, :, 3] = alpha * 255.0
        # Premultiply
        card_rgba[:, :, :3] *= alpha[:, :, np.newaxis]
        cards[comp] = card_rgba
        
    fps = 16
    total_frames = 78
    
    # Orange square, pink diamond, yellow circle
    moves = [
        (orange_square, 5, 22),
        (pink_diamond, 28, 45),
        (yellow_circle, 51, 68)
    ]
    
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=fps, codec='libx264', pixelformat='yuv420p')
    
    h, w, _ = img.shape
    
    for frame_idx in range(total_frames):
        frame = bg_canvas.copy().astype(np.float32)
        
        current_offsets = {orange_square: (0.0, 0.0), pink_diamond: (0.0, 0.0), yellow_circle: (0.0, 0.0)}
        
        for (comp, start_f, end_f) in moves:
            target_dx, target_dy = targets[comp]
            if frame_idx < start_f:
                current_offsets[comp] = (0.0, 0.0)
            elif frame_idx >= end_f:
                current_offsets[comp] = (target_dx, target_dy)
            else:
                t = (frame_idx - start_f) / float(end_f - start_f)
                ease_t = 0.5 - 0.5 * math.cos(math.pi * t)
                current_offsets[comp] = (target_dx * ease_t, target_dy * ease_t)
                
        # Draw all cards. Start with ones that are already placed, then the moving one, then the ones waiting.
        for comp in [yellow_circle, pink_diamond, orange_square]:
            dx, dy = current_offsets[comp]
            if dx == 0.0 and dy == 0.0:
                mask_2d = labels == comp
                frame[mask_2d] = img[mask_2d].astype(np.float32)
            else:
                card_rgba = cards[comp]
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                translated_rgba = cv2.warpAffine(card_rgba, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                
                alpha = translated_rgba[:, :, 3:4] / 255.0
                frame = frame * (1.0 - alpha) + translated_rgba[:, :, :3]
                
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    main()
