import numpy as np
from PIL import Image
import imageio
import os

def ease_in_out(t):
    return t * t * (3 - 2 * t)

def main():
    img = Image.open('/app/first_frame.png')
    base_arr = np.array(img)
    
    frames = []
    
    # 45 frames total
    
    original_sym5 = base_arr[464:604, 464:561].copy()
    sliding_block = base_arr[464:604, 569:981].copy()
    
    for f in range(45):
        frame = base_arr.copy()
        
        if f <= 18:
            # Fading phase
            alpha = f / 18.0
            faded_sym5 = original_sym5.astype(float) * (1 - alpha) + 255.0 * alpha
            frame[464:604, 464:561] = faded_sym5.astype(np.uint8)
        else:
            # Symbol 5 is completely white (background)
            frame[464:604, 464:561] = 255
            
            # Sliding phase
            if f < 20:
                t = 0.0
            elif f <= 40:
                t = (f - 20) / 20.0
            else:
                t = 1.0
                
            shift = int(round(105 * ease_in_out(t)))
            
            # Clear original position of moving symbols
            frame[464:604, 569:981] = 255
            # Paste at new position
            frame[464:604, 569 - shift : 981 - shift] = sliding_block
            
        frames.append(frame)
        
    os.makedirs('/app/output', exist_ok=True)
    # Write to video
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', macro_block_size=None, pixelformat='yuv420p')
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == '__main__':
    main()
