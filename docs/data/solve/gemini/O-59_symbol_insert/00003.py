import cv2
import numpy as np
import imageio

def get_symbol_rgba(patch):
    """Convert a white-background patch to RGBA."""
    rgba = np.zeros((patch.shape[0], patch.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = patch
    # Mask of non-white pixels
    mask = np.any(patch < 255, axis=-1)
    rgba[mask, 3] = 255
    return rgba

def overlay_image(bg, fg, x, y, alpha=1.0):
    """Overlay fg (RGBA) onto bg (BGR) at position (x, y) with given alpha."""
    h, w = fg.shape[:2]
    # Crop if out of bounds
    bg_h, bg_w = bg.shape[:2]
    
    y1, y2 = max(0, y), min(bg_h, y + h)
    x1, x2 = max(0, x), min(bg_w, x + w)
    
    fg_y1, fg_y2 = y1 - y, y2 - y
    fg_x1, fg_x2 = x1 - x, x2 - x
    
    if fg_y1 >= fg_y2 or fg_x1 >= fg_x2:
        return bg
        
    fg_crop = fg[fg_y1:fg_y2, fg_x1:fg_x2]
    bg_crop = bg[y1:y2, x1:x2]
    
    fg_alpha = (fg_crop[:, :, 3] / 255.0) * alpha
    
    for c in range(3):
        bg_crop[:, :, c] = fg_alpha * fg_crop[:, :, c] + (1 - fg_alpha) * bg_crop[:, :, c]
        
    return bg

def main():
    img = cv2.imread('/app/first_frame.png')
    
    # 1. Extract the new symbol from the reference panel
    x_panel, y_panel, w_panel, h_panel = 886, 17, 121, 121
    panel = img[y_panel:y_panel+h_panel, x_panel:x_panel+w_panel]
    green_mask = (panel[:,:,1] > 200) & (panel[:,:,0] < 50) & (panel[:,:,2] < 50)
    coords = np.argwhere(green_mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    new_symbol_bgr = panel[y_min:y_max+1, x_min:x_max+1]
    
    # Create RGBA for new symbol
    new_symbol_rgba = np.zeros((new_symbol_bgr.shape[0], new_symbol_bgr.shape[1], 4), dtype=np.uint8)
    new_symbol_rgba[:, :, :3] = new_symbol_bgr
    new_symbol_rgba[:, :, 3] = green_mask[y_min:y_max+1, x_min:x_max+1].astype(np.uint8) * 255

    # 2. Extract existing symbols that will move
    TARGET_IDX = 0
    moving_symbols = []
    for i in range(TARGET_IDX, 8):
        x_start = 45 + i * 105
        y_start = 465
        patch = img[y_start:y_start+94, x_start:x_start+94].copy()
        moving_symbols.append({
            'idx': i,
            'rgba': get_symbol_rgba(patch),
            'start_x': x_start,
            'start_y': y_start
        })
        
    # 3. Create the base background (erase the moving symbols)
    base_bg = img.copy()
    for sym in moving_symbols:
        x_start = sym['start_x']
        y_start = sym['start_y']
        base_bg[y_start:y_start+94, x_start:x_start+94] = (255, 255, 255)
        
    # 4. Generate frames
    frames = []
    total_frames = 24
    shift_frames = 10
    
    # Destination for new symbol
    dest_slot_x = 45 + TARGET_IDX * 105
    dest_slot_y = 465
    # Center new symbol in the 94x94 slot
    ns_h, ns_w = new_symbol_rgba.shape[:2]
    new_sym_x = dest_slot_x + (94 - ns_w) // 2
    new_sym_y = dest_slot_y + (94 - ns_h) // 2
    
    for f in range(total_frames):
        frame = base_bg.copy()
        
        # Phase 1 & 2: Moving symbols
        if f <= shift_frames:
            progress = f / shift_frames
        else:
            progress = 1.0
            
        for sym in moving_symbols:
            curr_x = int(sym['start_x'] + 105 * progress)
            frame = overlay_image(frame, sym['rgba'], curr_x, sym['start_y'])
            
        # Phase 2: New symbol fades in and slides down
        if f > shift_frames:
            phase2_progress = (f - shift_frames) / (total_frames - shift_frames - 1)
            # Slide down by 80 pixels
            start_y = new_sym_y - 80
            curr_y = int(start_y + 80 * phase2_progress)
            alpha = phase2_progress
            frame = overlay_image(frame, new_symbol_rgba, new_sym_x, curr_y, alpha)
            
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    # Save video
    import os
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', format='FFMPEG', macro_block_size=None, pixelformat='yuv420p')
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == '__main__':
    main()
