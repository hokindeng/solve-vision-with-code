import cv2
import numpy as np
import imageio

def get_rotated(sprite, mask, rot):
    if rot == 0:
        return sprite.copy(), mask.copy()
    elif rot == 1:
        return cv2.rotate(sprite, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    elif rot == 2:
        return cv2.flip(sprite, 1), cv2.flip(mask, 1)
    elif rot == 3:
        return cv2.rotate(sprite, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)

def main():
    img = cv2.imread('/app/first_frame.png')
    
    green = np.array([80, 175, 76])
    
    pacman_roi = img[84:213, 64:193].copy()
    mask_fg = cv2.bitwise_not(cv2.inRange(pacman_roi, green-5, green+5))
    
    clean_bg = img.copy()
    clean_bg[84:213, 64:193] = green
    
    path = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (2, 1), (1, 1), (0, 1), (0, 2), (0, 3), (1, 3), (1, 2), (2, 2), (2, 3), (3, 3)]
    
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, format='FFMPEG', mode='I', codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    for i in range(91):
        if i < 4:
            r, c = path[0]
            curr_X = c * 256 + 64
            curr_Y = r * 256 + 84
            rot = 0
            
            # for frame 0, we can ensure it's exactly the first frame
            if i == 0:
                writer.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                continue
        elif i >= 88:
            r, c = path[-1]
            curr_X = c * 256 + 64
            curr_Y = r * 256 + 84
            dr = path[-1][0] - path[-2][0]
            dc = path[-1][1] - path[-2][1]
            if dc > 0: rot = 0
            elif dr > 0: rot = 1
            elif dc < 0: rot = 2
            elif dr < 0: rot = 3
        else:
            frame_move = i - 4
            seg = frame_move // 6
            f = frame_move % 6
            alpha = f / 6.0
            
            r1, c1 = path[seg]
            r2, c2 = path[seg+1]
            
            curr_r = r1 + (r2 - r1) * alpha
            curr_c = c1 + (c2 - c1) * alpha
            
            curr_X = curr_c * 256 + 64
            curr_Y = curr_r * 256 + 84
            
            dc = c2 - c1
            dr = r2 - r1
            if dc > 0: rot = 0
            elif dr > 0: rot = 1
            elif dc < 0: rot = 2
            elif dr < 0: rot = 3
            
        frame = clean_bg.copy()
        s, m = get_rotated(pacman_roi, mask_fg, rot)
        
        X = int(round(curr_X))
        Y = int(round(curr_Y))
        
        roi = frame[Y:Y+s.shape[0], X:X+s.shape[1]]
        np.copyto(roi, s, where=(m[:,:,None]>0))
        
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    writer.close()
    print("Done")

if __name__ == '__main__':
    main()
