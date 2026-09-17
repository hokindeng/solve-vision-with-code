import cv2
import numpy as np
import imageio
import os

def generate_video():
    first_frame_path = '/app/first_frame.png'
    output_dir = '/app/output'
    output_path = os.path.join(output_dir, 'video.mp4')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    img = cv2.imread(first_frame_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Precompute masks and colors
    # Colors in RGB
    color_green = [126, 196, 138] # BGR: 138, 196, 126
    color_blue = [111, 169, 255] # BGR: 255, 169, 111
    color_pink = [255, 140, 170] # BGR: 170, 140, 255
    
    white_mask = np.all(img_rgb == [255, 255, 255], axis=-1)
    
    curves = [
        # Green curve
        {'p0': (415, 280), 'p1': (631, 150), 'p2': (847, 286), 'color': color_green, 'start_f': 1, 'end_f': 14},
        # Blue curve
        {'p0': (290, 475), 'p1': (436, 680), 'p2': (582, 490), 'color': color_blue, 'start_f': 15, 'end_f': 29},
        # Pink curve
        {'p0': (190, 756), 'p1': (374, 980), 'p2': (559, 812), 'color': color_pink, 'start_f': 30, 'end_f': 44}
    ]
    
    num_frames = 48
    fps = 16
    thickness = 8
    
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p')
    
    # We will accumulate the drawn curves on a persistent canvas
    # to avoid re-drawing the complete curves every frame.
    # Actually, re-drawing is fast enough, let's just redraw for simplicity.
    
    for f in range(num_frames):
        frame_rgb = img_rgb.copy()
        
        # Draw each curve up to its current progress
        for curve in curves:
            if f < curve['start_f']:
                continue # not started yet
                
            progress = (f - curve['start_f'] + 1) / (curve['end_f'] - curve['start_f'] + 1)
            progress = min(max(progress, 0.0), 1.0)
            
            if progress > 0:
                temp = np.zeros_like(img_rgb)
                
                # Get points up to progress
                num_points = max(2, int(500 * progress))
                
                pts = []
                for t in np.linspace(0, progress, num_points):
                    x = int(round((1-t)**2 * curve['p0'][0] + 2*(1-t)*t * curve['p1'][0] + t**2 * curve['p2'][0]))
                    y = int(round((1-t)**2 * curve['p0'][1] + 2*(1-t)*t * curve['p1'][1] + t**2 * curve['p2'][1]))
                    pts.append((x,y))
                    
                for i in range(len(pts)-1):
                    cv2.line(temp, pts[i], pts[i+1], curve['color'], thickness, cv2.LINE_AA)
                    
                max_c_idx = np.argmax(curve['color'])
                max_c_val = curve['color'][max_c_idx]
                
                # Avoid division by zero just in case
                if max_c_val > 0:
                    alpha = temp[:, :, max_c_idx].astype(np.float32) / max_c_val
                else:
                    alpha = np.zeros(temp.shape[:2], dtype=np.float32)
                alpha = np.clip(alpha, 0, 1)
                
                for c in range(3):
                    blended = temp[:, :, c].astype(np.float32) + 255.0 * (1.0 - alpha)
                    apply_mask = white_mask & (alpha > 0)
                    frame_rgb[apply_mask, c] = np.clip(blended[apply_mask], 0, 255).astype(np.uint8)
                    
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    generate_video()
