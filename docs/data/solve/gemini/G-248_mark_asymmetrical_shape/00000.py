import cv2
import numpy as np
import os
import subprocess
import tempfile

def get_symmetry_score(contour, thresh_img):
    x, y, w, h = cv2.boundingRect(contour)
    roi = thresh_img[y:y+h, x:x+w].copy()
    
    M = cv2.moments(roi)
    if M['m00'] == 0: return 0
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']
    
    size = max(w, h) * 2
    canvas = np.zeros((size, size), dtype=np.uint8)
    dx = int(size/2 - cx)
    dy = int(size/2 - cy)
    canvas[dy:dy+h, dx:dx+w] = roi
    
    min_diff = float('inf')
    for angle in range(0, 180, 5):
        M_rot = cv2.getRotationMatrix2D((size/2, size/2), angle, 1)
        rot = cv2.warpAffine(canvas, M_rot, (size, size))
        flip = cv2.flip(rot, 1)
        diff = np.sum(cv2.absdiff(rot, flip) > 0)
        if diff < min_diff:
            min_diff = diff
            
    return min_diff / np.sum(canvas > 0)

def main():
    os.makedirs('/app/output', exist_ok=True)
    
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise FileNotFoundError("Could not read /app/first_frame.png")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold assuming background is white
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    scores = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 100: continue
        score = get_symmetry_score(cnt, thresh)
        scores.append((score, cnt))
        
    if not scores:
        raise ValueError("No shapes found")
        
    scores.sort(key=lambda x: x[0], reverse=True)
    asym_cnt = scores[0][1]
    
    (cx, cy), radius = cv2.minEnclosingCircle(asym_cnt)
    
    num_frames = 16
    draw_radius = int(radius) + 20
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(num_frames):
            frame = img.copy()
            if i > 0:
                end_angle = 360 * i / (num_frames - 1)
                cv2.ellipse(frame, (int(cx), int(cy)), (draw_radius, draw_radius), 
                            0, 0, end_angle, (0, 0, 255), 8, cv2.LINE_AA)
            cv2.imwrite(os.path.join(temp_dir, f'frame_{i:02d}.png'), frame)
            
        cmd = [
            'ffmpeg', '-y',
            '-framerate', '16',
            '-i', os.path.join(temp_dir, 'frame_%02d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-s', f'{img.shape[1]}x{img.shape[0]}',
            '/app/output/video.mp4'
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
if __name__ == '__main__':
    main()
