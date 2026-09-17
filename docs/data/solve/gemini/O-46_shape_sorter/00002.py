import cv2
import numpy as np
import imageio
import os

def ease_in_out(t):
    return t * t * (3 - 2 * t)

def main():
    # Load first frame
    first_frame_path = '/app/first_frame.png'
    first_frame_bgr = cv2.imread(first_frame_path)
    if first_frame_bgr is None:
        raise FileNotFoundError(f"Could not find {first_frame_path}")
    
    first_frame = cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2RGB)
    
    bg_color = np.array([248, 250, 252]) # RGB
    
    sequence = ['yellow', 'red', 'orange', 'cyan', 'pink']
    
    # Exact RGB colors of the shapes
    colors = {
        'yellow': np.array([250, 204, 21]),
        'red': np.array([248, 113, 113]),
        'orange': np.array([251, 146, 60]),
        'cyan': np.array([34, 211, 238]),
        'pink': np.array([244, 114, 182])
    }
    
    # Target offsets computed
    offsets = {
        'yellow': (493, 8),
        'red': (498, 0),
        'orange': (488, -6),
        'cyan': (490, 2),
        'pink': (491, -1)
    }
    
    os.makedirs('/app/output', exist_ok=True)
    out_path = '/app/output/video.mp4'
    
    # Video parameters
    fps = 16
    total_frames = 77
    
    writer = imageio.get_writer(out_path, fps=fps, codec='libx264', format='FFMPEG', pixelformat='yuv420p')
    
    for f in range(total_frames):
        img_current = first_frame.copy()
        
        # Erase all shapes from their initial positions
        for name in sequence:
            color = colors[name]
            mask = np.all(first_frame == color, axis=-1)
            img_current[mask] = bg_color
            
        moving_idx = min(f // 15, 4)
        
        # Draw stationary shapes (either at start or end)
        for j, name in enumerate(sequence):
            if j == moving_idx:
                continue
                
            color = colors[name]
            mask = np.all(first_frame == color, axis=-1)
            dx_full, dy_full = offsets[name]
            
            if j < moving_idx:
                dx, dy = dx_full, dy_full
            else:
                dx, dy = 0, 0
                
            y_idx, x_idx = np.where(mask)
            y_shift = y_idx + dy
            x_shift = x_idx + dx
            
            img_current[y_shift, x_shift] = color
            
        # Draw the moving shape on top
        if moving_idx < 5:
            name = sequence[moving_idx]
            start_f = moving_idx * 15
            end_f = (moving_idx + 1) * 15
            
            t = (f - start_f) / float(end_f - start_f)
            t = max(0.0, min(1.0, t))
            t = ease_in_out(t)
            
            dx_full, dy_full = offsets[name]
            dx = int(round(dx_full * t))
            dy = int(round(dy_full * t))
            
            color = colors[name]
            mask = np.all(first_frame == color, axis=-1)
            y_idx, x_idx = np.where(mask)
            y_shift = y_idx + dy
            x_shift = x_idx + dx
            
            img_current[y_shift, x_shift] = color
            
        writer.append_data(img_current)
        
    writer.close()
    print("Video saved to", out_path)

if __name__ == "__main__":
    main()
