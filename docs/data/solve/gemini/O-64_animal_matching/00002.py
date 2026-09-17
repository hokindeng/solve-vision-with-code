import cv2
import numpy as np
import subprocess
import os

def main():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise ValueError("Could not read /app/first_frame.png")
        
    bg_color = img[0, 0]
    diff = np.abs(img.astype(np.int32) - bg_color.astype(np.int32))
    base_mask = (np.sum(diff, axis=2) > 0).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(base_mask)

    left_comps = []
    right_comps = []
    
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area > 100 and w < 500:
            if x < 512:
                left_comps.append((i, x, y, w, h))
            else:
                right_comps.append((i, x, y, w, h))
                
    matches = []
    for l in left_comps:
        best = None
        best_diff = float('inf')
        for r in right_comps:
            diff_val = abs(l[3] - r[3]) + abs(l[4] - r[4])
            if diff_val < best_diff:
                best_diff = diff_val
                best = r
        
        cx_l = l[1] + l[3]/2.0
        cy_l = l[2] + l[4]/2.0
        cx_r = best[1] + best[3]/2.0
        cy_r = best[2] + best[4]/2.0
        
        dx = cx_r - cx_l
        dy = cy_r - cy_l
        
        mask = (labels == l[0]).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(mask)
        cv2.drawContours(filled_mask, contours, -1, 1, -1)
        
        matches.append({
            'label': l[0],
            'dx': dx,
            'dy': dy,
            'mask': filled_mask
        })
        
    bg_img = img.copy()
    for m in matches:
        bg_img[m['mask'] == 1] = bg_color
        
    os.makedirs('/app/output', exist_ok=True)
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{img.shape[1]}x{img.shape[0]}',
        '-pix_fmt', 'bgr24',
        '-r', '16',
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '/app/output/video.mp4'
    ]
    
    # Do not capture stderr so it prints to console or gets discarded, avoiding pipe blocking.
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    
    for t in range(64):
        alpha = t / 63.0
        frame = bg_img.copy()
        
        for m in matches:
            cur_dx = int(round(m['dx'] * alpha))
            cur_dy = int(round(m['dy'] * alpha))
            
            mask = m['mask']
            ys, xs = np.where(mask == 1)
            
            frame[ys + cur_dy, xs + cur_dx] = img[ys, xs]
            
        process.stdin.write(frame.tobytes())
        
    process.stdin.close()
    process.wait()

if __name__ == '__main__':
    main()
