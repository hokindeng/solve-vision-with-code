import cv2
import numpy as np
import os
import imageio

def get_unique_shape_center(img):
    img_f = img.astype(np.float32)
    bg_color = img_f[0, 0]
    
    diff = np.linalg.norm(img_f - bg_color, axis=2)
    mask = np.uint8(diff > 10) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shapes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:
            shapes.append({
                'cnt': cnt, 
                'x': x, 'y': y, 'w': w, 'h': h, 
                'area': cv2.contourArea(cnt)
            })
            
    if not shapes:
        return None, None
        
    if len(shapes) == 1:
        s = shapes[0]
        return (s['x'] + s['w']//2, s['y'] + s['h']//2), int(max(s['w'], s['h']) * 0.75)

    n = len(shapes)
    dist_sums = np.zeros(n)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                sd = cv2.matchShapes(shapes[i]['cnt'], shapes[j]['cnt'], cv2.CONTOURS_MATCH_I1, 0)
                a1, a2 = shapes[i]['area'], shapes[j]['area']
                ad = abs(a1 - a2) / max(a1, a2) if max(a1, a2) > 0 else 0
                dist_sums[i] += (sd + ad)
                
    unique_idx = np.argmax(dist_sums)
    unique_shape = shapes[unique_idx]
    
    cx = unique_shape['x'] + unique_shape['w'] // 2
    cy = unique_shape['y'] + unique_shape['h'] // 2
    
    diag = np.sqrt(unique_shape['w']**2 + unique_shape['h']**2)
    radius = int(diag / 2 * 1.2)
    
    return (cx, cy), radius

def generate_video():
    img_path = '/app/first_frame.png'
    output_path = '/app/output/video.mp4'
    os.makedirs('/app/output', exist_ok=True)
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read {img_path}")
        return
        
    center, radius = get_unique_shape_center(img)
    if center is None:
        print("Could not find unique shape")
        return
        
    writer = imageio.get_writer(output_path, fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=1)
    
    total_frames = 60
    start_draw = 10
    end_draw = 50
    
    for i in range(total_frames):
        frame = img.copy()
        
        if i >= start_draw:
            if i >= end_draw:
                angle = 360
            else:
                progress = (i - start_draw) / (end_draw - start_draw)
                angle = int(progress * 360)
                
            if angle > 0:
                cv2.ellipse(frame, center, (radius, radius), -90, 0, angle, (0, 0, 255), 5, cv2.LINE_AA)
                
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()
    print("Video generated successfully.")

if __name__ == '__main__':
    generate_video()
