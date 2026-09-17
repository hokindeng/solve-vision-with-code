import cv2
import numpy as np
import imageio

def get_new_symbol(img):
    ref_crop = img[18:138, 887:1007]
    mask = (ref_crop[:,:,0] == 255) & (ref_crop[:,:,1] == 0) & (ref_crop[:,:,2] == 0)
    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    star = ref_crop[y_min:y_max+1, x_min:x_max+1]

    new_sym = np.full((97, 97, 3), 255, dtype=np.uint8)
    new_sym[0, :] = 170
    new_sym[-1, :] = 170
    new_sym[:, 0] = 170
    new_sym[:, -1] = 170

    h, w = star.shape[:2]
    y_offset = (97 - h) // 2
    x_offset = (97 - w) // 2

    for y in range(h):
        for x in range(w):
            if np.array_equal(star[y, x], [255, 0, 0]):
                new_sym[y_offset+y, x_offset+x] = [255, 0, 0]
    return new_sym

def ease_in_out(t):
    return t * t * (3.0 - 2.0 * t)

def main():
    img = cv2.imread('/app/first_frame.png')
    new_sym = get_new_symbol(img)
    sym5 = img[464:464+97, 674:674+97].copy()
    
    # Base image for drop phase (symbol 5 already moved)
    base_drop = img.copy()
    base_drop[464:464+97, 674:674+97] = 255
    base_drop[464:464+97, 779:779+97] = sym5
    
    frames = []
    
    for f in range(24):
        if f == 0:
            frame = img.copy()
        elif f <= 9:
            # sliding phase
            frame = img.copy()
            frame[464:464+97, 674:674+97] = 255
            
            t = f / 9.0
            t_ease = ease_in_out(t)
            x_curr = 674 + int(105 * t_ease)
            frame[464:464+97, x_curr:x_curr+97] = sym5
        elif f <= 14:
            # fade in phase above the gap
            frame = base_drop.copy()
            t = (f - 9) / 5.0
            
            y_curr = 364
            alpha = t
            
            bg_crop = frame[y_curr:y_curr+97, 674:674+97].astype(float)
            new_sym_f = new_sym.astype(float)
            blended = (new_sym_f * alpha + bg_crop * (1 - alpha)).astype(np.uint8)
            frame[y_curr:y_curr+97, 674:674+97] = blended
        elif f <= 22:
            # drop phase
            frame = base_drop.copy()
            t = (f - 14) / 8.0
            t_ease = ease_in_out(t)
            
            y_curr = 364 + int(100 * t_ease)
            frame[y_curr:y_curr+97, 674:674+97] = new_sym
        else:
            # final hold
            frame = base_drop.copy()
            frame[464:464+97, 674:674+97] = new_sym
            
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == '__main__':
    main()
