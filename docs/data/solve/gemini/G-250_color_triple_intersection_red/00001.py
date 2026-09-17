import cv2
import numpy as np
import os

def create_video():
    img = cv2.imread('/app/first_frame.png')
    
    # Create mask for flood fill
    # We want to fill the region containing (523, 488)
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    
    seed_pt = (523, 488)
    
    # Perform flood fill on a copy to get the filled region
    filled_img = img.copy()
    cv2.floodFill(filled_img, mask, seed_pt, (0, 0, 255), loDiff=(5, 5, 5), upDiff=(5, 5, 5))
    
    # The mask returned by floodFill has 1s where the image was filled.
    # The mask is 2 pixels larger in each dimension.
    fill_mask = mask[1:h+1, 1:w+1]
    
    os.makedirs('/app/output', exist_ok=True)
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    raw_path = '/app/output/video_raw.mp4'
    out = cv2.VideoWriter(raw_path, fourcc, 16, (w, h))
    
    frames = 60
    for i in range(frames):
        alpha = i / (frames - 1)
        
        # We can blend the color.
        # original is white (255, 255, 255)
        # target is red (0, 0, 255) (BGR)
        
        frame = img.copy()
        
        # Calculate intermediate color for the filled region
        b = int(255 * (1 - alpha) + 0 * alpha)
        g = int(255 * (1 - alpha) + 0 * alpha)
        r = int(255 * (1 - alpha) + 255 * alpha)
        
        frame[fill_mask == 1] = [b, g, r]
        
        out.write(frame)
        
    out.release()
    
    # convert to yuv420p and set h264
    final_path = '/app/output/video.mp4'
    os.system(f'ffmpeg -y -i {raw_path} -c:v libx264 -pix_fmt yuv420p -r 16 {final_path}')
    os.remove(raw_path)

if __name__ == '__main__':
    create_video()
