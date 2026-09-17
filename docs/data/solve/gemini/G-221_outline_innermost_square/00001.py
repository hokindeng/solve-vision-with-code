import imageio.v2 as imageio
import numpy as np

def main():
    img = imageio.imread('/app/first_frame.png')
    
    # Identify the innermost square boundaries
    mid_row = img[img.shape[0] // 2, :]
    changes = np.where(np.any(mid_row[:-1] != mid_row[1:], axis=1))[0]
    
    # changes mark the transitions between the background and concentric squares.
    if len(changes) >= 2:
        mid_idx = len(changes) // 2
        L = changes[mid_idx - 1] + 1
        R = changes[mid_idx]
    else:
        # Fallback coordinates for robust execution based on initial frame analysis
        L, R = 224, 800
        
    T = 16 # outline thickness
    
    total_frames = 85
    start_frame = 10
    end_frame = 75
    trace_frames = end_frame - start_frame
    
    top_len = R - L + 1
    right_len = R - (L + T) + 1
    bottom_len = R - T - L + 1
    left_len = R - T - (L + T) + 1
    
    total_len = top_len + right_len + bottom_len + left_len
    
    def draw_progress(frame, progress):
        d = int(progress * total_len)
        
        # Draw Top Edge
        if d > 0:
            draw_len = min(d, top_len)
            frame[L:L+T, L:L+draw_len] = [0, 0, 255] # Pure Blue in RGB
            d -= draw_len
            
        # Draw Right Edge
        if d > 0:
            draw_len = min(d, right_len)
            frame[L+T:L+T+draw_len, R-T+1:R+1] = [0, 0, 255]
            d -= draw_len
            
        # Draw Bottom Edge (right to left)
        if d > 0:
            draw_len = min(d, bottom_len)
            x_start = R - T + 1
            x_end = x_start - draw_len
            frame[R-T+1:R+1, x_end:x_start] = [0, 0, 255]
            d -= draw_len
            
        # Draw Left Edge (bottom to top)
        if d > 0:
            draw_len = min(d, left_len)
            y_start = R - T + 1
            y_end = y_start - draw_len
            frame[y_end:y_start, L:L+T] = [0, 0, 255]

    writer = imageio.get_writer(
        '/app/output/video.mp4', 
        fps=16, 
        codec='libx264', 
        quality=9, 
        pixelformat='yuv420p', 
        macro_block_size=None
    )
    
    for i in range(total_frames):
        frame = img.copy()
        if i >= start_frame:
            progress = min(1.0, (i - start_frame) / trace_frames)
            draw_progress(frame, progress)
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    main()
