import cv2
import numpy as np
import imageio
import os

def solve():
    img_path = '/app/first_frame.png'
    out_path = '/app/output/video.mp4'
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read {img_path}")
        
    H, W = img.shape[:2]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    objects = []
    for c in contours:
        if cv2.contourArea(c) > 100:
            M = cv2.moments(c)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = 0, 0
            objects.append({
                'contour': c,
                'cx': cx,
                'cy': cy
            })
            
    # Systematically order from left to right (and top to bottom for ties)
    objects.sort(key=lambda o: (o['cx'], o['cy']))
    
    total_objects = len(objects)
    
    def get_opacity(start_frame, duration, t):
        if t < start_frame:
            return 0.0
        elif t >= start_frame + duration:
            return 1.0
        else:
            return (t - start_frame) / duration

    frames = []
    total_frames = 55
    
    # 7 objects. Paced sequentially over the available time.
    # We fade in each object's border over `fade_duration` frames.
    fade_duration = 6
    
    text = f"Count: {total_objects}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3.0
    thickness = 8
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    tcx = (W - tw) // 2
    tcy = (H + th) // 2
    
    for t in range(total_frames):
        frame = img.copy().astype(np.float32)
        
        # Sequentially highlight objects
        for i, obj in enumerate(objects):
            start_f = i * fade_duration
            opacity = get_opacity(start_f, fade_duration, t)
            
            if opacity > 0:
                overlay = np.zeros((H, W, 4), dtype=np.uint8)
                color = (0, 0, 255, int(255 * opacity))
                cv2.drawContours(overlay, [obj['contour']], -1, color, 5, cv2.LINE_AA)
                
                alpha = overlay[:, :, 3] / 255.0
                for c_idx in range(3):
                    frame[:, :, c_idx] = frame[:, :, c_idx] * (1 - alpha) + overlay[:, :, c_idx] * alpha
                    
        # Fade in text at the end
        text_start_f = total_objects * fade_duration
        text_opacity = get_opacity(text_start_f, fade_duration, t)
        
        if text_opacity > 0:
            text_overlay = np.zeros((H, W, 4), dtype=np.uint8)
            
            # White outline
            outline_color = (255, 255, 255, int(255 * text_opacity))
            cv2.putText(text_overlay, text, (tcx, tcy), font, font_scale, outline_color, thickness + 10, cv2.LINE_AA)
            
            # Black body
            body_color = (0, 0, 0, int(255 * text_opacity))
            cv2.putText(text_overlay, text, (tcx, tcy), font, font_scale, body_color, thickness, cv2.LINE_AA)
            
            t_alpha = text_overlay[:, :, 3] / 255.0
            for c_idx in range(3):
                frame[:, :, c_idx] = frame[:, :, c_idx] * (1 - t_alpha) + text_overlay[:, :, c_idx] * t_alpha
                
        # Clip and convert
        frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
        frame_rgb = cv2.cvtColor(frame_uint8, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    # Write H.264 mp4
    with imageio.get_writer(out_path, fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None) as writer:
        for f in frames:
            writer.append_data(f)

if __name__ == '__main__':
    solve()
