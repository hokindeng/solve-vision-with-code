import cv2
import numpy as np
import imageio
import os

def get_bezier_curve(p0, p1, p2, num_points=500):
    t = np.linspace(0, 1, num_points)
    x = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
    y = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
    return np.vstack((x, y)).T.astype(np.int32)

def main():
    img = cv2.imread('/app/first_frame.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    bg_mask = np.all(img == [255, 255, 255], axis=-1)
    shape_mask = ~bg_mask

    colors_bgr = {
        'P': (255, 178, 170), # This is BGR or RGB? The prompt unique colors were RGB.
        # Wait, the cv2.imread loads BGR, but I converted to RGB.
        # So I should use RGB colors!
    }

    # Let's dynamically find the colors and their centers from the image.
    unique_colors, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
    # Sort by count descending
    color_counts = sorted(zip(counts, unique_colors), key=lambda x: x[0], reverse=True)
    
    # bg is the most frequent
    bg_color = color_counts[0][1]
    
    shapes_info = []
    for count, color in color_counts[1:4]:
        # Create mask for this color
        c_mask = np.all(img == color, axis=-1).astype(np.uint8) * 255
        contours, _ = cv2.findContours(c_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centers = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))
        
        centers.sort(key=lambda p: p[0])
        shapes_info.append({
            'color': tuple(int(x) for x in color),
            'centers': centers
        })
        
    # We should order the shapes based on their y position to avoid overlaps.
    # Top shapes arch up, bottom shapes arch down, middle can arch up/down.
    shapes_info.sort(key=lambda s: s['centers'][0][1]) # Sort by y-coord
    
    # Now top to bottom
    curves = []
    for i, info in enumerate(shapes_info):
        p0, p2 = info['centers']
        color = info['color']
        
        # Decide arch direction
        if i == 0:
            # Top-most, arch up
            p1 = ((p0[0]+p2[0])//2, min(p0[1], p2[1]) - 150)
        elif i == 2:
            # Bottom-most, arch down
            p1 = ((p0[0]+p2[0])//2, max(p0[1], p2[1]) + 150)
        else:
            # Middle, arch up or down. Arching up slightly is fine.
            p1 = ((p0[0]+p2[0])//2, min(p0[1], p2[1]) - 120)
            
        pts = get_bezier_curve(p0, p1, p2, 500)
        
        # Clip to background
        valid_indices = []
        for j, (x, y) in enumerate(pts):
            if not shape_mask[y, x]:
                valid_indices.append(j)
                
        if valid_indices:
            start_idx = valid_indices[0]
            end_idx = valid_indices[-1]
            clipped_pts = pts[start_idx:end_idx+1]
        else:
            clipped_pts = pts
            
        curves.append({
            'color': color,
            'pts': clipped_pts
        })
        
    frames = []
    total_frames = 48
    fps = 16
    
    # We will draw them sequentially.
    # We have 3 curves. Let's use 14 frames per curve.
    frames_per_curve = 14
    
    # Pre-calculate what is fully drawn up to the current frame.
    for f in range(total_frames):
        frame = img.copy()
        
        for i, curve_info in enumerate(curves):
            start_f = i * frames_per_curve
            end_f = start_f + frames_per_curve
            
            if f >= start_f:
                progress = min(1.0, (f - start_f) / float(frames_per_curve))
                pts = curve_info['pts']
                num_pts = int(progress * len(pts))
                
                if num_pts > 1:
                    draw_pts = pts[:num_pts].reshape((-1, 1, 2))
                    cv2.polylines(frame, [draw_pts], isClosed=False, color=curve_info['color'], thickness=8, lineType=cv2.LINE_AA)
                    
        # Restore shapes so lines are perfectly underneath without bleeding
        frame[shape_mask] = img[shape_mask]
        
        frames.append(frame)
        
    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=fps, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    main()
