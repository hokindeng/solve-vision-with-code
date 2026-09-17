import cv2
import numpy as np
import imageio

def get_bezier_curve(p1, p2, p_control, num_points=1000):
    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, 2))
    curve[:, 0] = (1 - t)**2 * p1[0] + 2 * (1 - t) * t * p_control[0] + t**2 * p2[0]
    curve[:, 1] = (1 - t)**2 * p1[1] + 2 * (1 - t) * t * p_control[1] + t**2 * p2[1]
    return curve

def main():
    img = cv2.imread('/app/first_frame.png')
    bg_color = img[0, 0]
    
    unique_colors = np.unique(img.reshape(-1, 3), axis=0)
    
    tasks = []
    
    for c in unique_colors:
        if np.array_equal(c, bg_color):
            continue
            
        mask = cv2.inRange(img, c, c)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centers = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                centers.append((cx, cy))
                
        # Sort left to right
        centers.sort(key=lambda p: p[0])
        
        if len(centers) >= 2:
            p1 = np.array(centers[0])
            p2 = np.array(centers[-1])
            
            M = (p1 + p2) / 2.0
            D = p2 - p1
            N = np.array([-D[1], D[0]], dtype=float)
            N /= np.linalg.norm(N)
            C = M + N * 100
            
            curve = get_bezier_curve(p1, p2, C, num_points=2000)
            
            tasks.append({
                'color': c,
                'curve': curve
            })
            
    frames = []
    num_frames = 48
    
    # Pre-calculate shape mask (where original image is not background)
    shape_mask = cv2.inRange(img, bg_color, bg_color)
    shape_mask = cv2.bitwise_not(shape_mask)
    
    for i in range(num_frames):
        frame = img.copy()
        t_end = i / (num_frames - 1) if num_frames > 1 else 1.0
        
        curve_canvas = np.zeros_like(img)
        curve_canvas[:] = bg_color
        
        for task in tasks:
            c = task['color']
            curve = task['curve']
            pts_to_draw = int(t_end * len(curve))
            
            if pts_to_draw > 1:
                pts = np.round(curve[:pts_to_draw]).astype(np.int32)
                cv2.polylines(curve_canvas, [pts], isClosed=False, color=c.tolist(), thickness=10, lineType=cv2.LINE_AA)
                
        # Apply curves to frame, but keep original shapes on top
        # Where shape_mask is 0 (background), we use curve_canvas
        # Where shape_mask is 255 (shapes), we use original frame
        mask_bool = shape_mask > 0
        frame_bg = curve_canvas.copy()
        frame_bg[mask_bool] = frame[mask_bool]
        
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame_bg, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    import os
    os.makedirs('/app/output', exist_ok=True)
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', pixelformat='yuv420p', quality=10)
    
if __name__ == '__main__':
    main()
