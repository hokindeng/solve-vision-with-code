import cv2
import numpy as np
import imageio
import os

def create_box_mask(w, h, patch):
    c_size = 9
    res = np.zeros((h, w), dtype=np.uint8)
    res[0:c_size, 0:c_size] = patch[0:c_size, 0:c_size]
    res[0:c_size, w-c_size:w] = patch[0:c_size, -c_size:]
    res[h-c_size:h, 0:c_size] = patch[-c_size:, 0:c_size]
    res[h-c_size:h, w-c_size:w] = patch[-c_size:, -c_size:]
    
    if w > 2*c_size:
        res[0:c_size, c_size:w-c_size] = np.repeat(patch[0:c_size, c_size:c_size+1], w-2*c_size, axis=1)
        res[h-c_size:h, c_size:w-c_size] = np.repeat(patch[-c_size:, c_size:c_size+1], w-2*c_size, axis=1)
    if h > 2*c_size:
        res[c_size:h-c_size, 0:c_size] = np.repeat(patch[c_size:c_size+1, 0:c_size], h-2*c_size, axis=0)
        res[c_size:h-c_size, w-c_size:w] = np.repeat(patch[c_size:c_size+1, -c_size:], h-2*c_size, axis=0)
        
    return res

def main():
    img = cv2.imread('/app/first_frame.png')
    
    color = np.array([70, 140, 70])
    mask = cv2.inRange(img, color, color)
    box_patch = mask[331:693, 80:442]
    
    clean_img = img.copy()
    clean_img[mask > 0] = [255, 255, 255]
    
    x1, y1, w1, h1 = 80, 331, 362, 362
    x2, y2, w2, h2 = 567, 307, 365, 178
    
    num_frames = 25
    fps = 16
    
    os.makedirs('/app/output', exist_ok=True)
    out_path = '/app/output/video.mp4'
    writer = imageio.get_writer(out_path, fps=fps, codec='libx264', macro_block_size=None, pixelformat='yuv420p')
    
    for i in range(num_frames):
        t = i / (num_frames - 1)
        
        cur_x = int(round(x1 + (x2 - x1) * t))
        cur_y = int(round(y1 + (y2 - y1) * t))
        cur_w = int(round(w1 + (w2 - w1) * t))
        cur_h = int(round(h1 + (h2 - h1) * t))
        
        frame = clean_img.copy()
        
        cur_mask = create_box_mask(cur_w, cur_h, box_patch)
        
        # Color the mask area on the frame
        frame_roi = frame[cur_y:cur_y+cur_h, cur_x:cur_x+cur_w]
        frame_roi[cur_mask > 0] = [70, 140, 70]
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    main()
