import cv2
import numpy as np
import imageio

def ease_in_out(t):
    return t * t * (3.0 - 2.0 * t)

def main():
    img = cv2.imread('/app/first_frame.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    bg_base = img.copy()
    bg_base[:, 517:] = 255
    
    target_sym = img[:, 517:622].copy()
    sliding_syms = img[:, 622:].copy()
    
    TOTAL_FRAMES = 45
    FADE_END = 20
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', macro_block_size=None, pixelformat='yuv420p')
    
    for f in range(TOTAL_FRAMES):
        frame = bg_base.copy()
        
        if f <= FADE_END:
            t = f / FADE_END
            alpha = 1.0 - t
            if alpha == 1.0:
                blended = target_sym.copy()
            elif alpha == 0.0:
                blended = np.full_like(target_sym, 255)
            else:
                white = np.full_like(target_sym, 255)
                blended = cv2.addWeighted(target_sym, alpha, white, 1 - alpha, 0)
            frame[:, 517:622] = blended
            shift = 0
        else:
            progress = (f - FADE_END) / (TOTAL_FRAMES - 1 - FADE_END)
            shift = int(round(105 * ease_in_out(progress)))
            
        x_start = 622 - shift
        x_end = x_start + sliding_syms.shape[1]
        
        frame[:, x_start:x_end] = sliding_syms
        
        writer.append_data(frame)
        
    writer.close()

if __name__ == '__main__':
    main()
