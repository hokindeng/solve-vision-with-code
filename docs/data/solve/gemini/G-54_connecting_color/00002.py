import cv2
import numpy as np
import imageio
import os

def get_bezier_points(p0, p1, p2, num_points=500):
    pts = []
    for u in np.linspace(0, 1, num_points):
        p = (1-u)**2 * p0 + 2*(1-u)*u * p1 + u**2 * p2
        pts.append(p)
    return np.array(pts, dtype=np.int32)

def main():
    orig_img = cv2.imread('/app/first_frame.png')
    h, w = orig_img.shape[:2]

    # Colors as found in the image (BGR)
    cyan = (251, 219, 72)
    purple = (255, 178, 170)

    # Calculate Centers programmatically
    def get_centers(color):
        mask = cv2.inRange(orig_img, np.array(color), np.array(color))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centers.append((cx, cy))
        # Sort by x coordinate (left to right)
        centers.sort(key=lambda pt: pt[0])
        return centers

    cyan_centers = get_centers(cyan)
    purple_centers = get_centers(purple)

    c_p0 = np.array(cyan_centers[0])
    c_p2 = np.array(cyan_centers[1])
    c_p1 = np.array([(c_p0[0]+c_p2[0])//2, max(c_p0[1], c_p2[1]) + 150])

    p_p0 = np.array(purple_centers[0])
    p_p2 = np.array(purple_centers[1])
    p_p1 = np.array([(p_p0[0]+p_p2[0])//2, min(p_p0[1], p_p2[1]) - 150])

    c_pts = get_bezier_points(c_p0, c_p1, c_p2)
    p_pts = get_bezier_points(p_p0, p_p1, p_p2)

    num_frames = 48
    
    # Create mask of shapes (non-white pixels)
    mask = (orig_img != 255).any(axis=-1)

    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')

    for i in range(num_frames):
        # Base frame is white
        frame = np.full((h, w, 3), 255, dtype=np.uint8)
        
        t = i / (num_frames - 1)
        idx = int(t * (len(c_pts) - 1))
        
        if idx > 0:
            c_sub = c_pts[:idx+1]
            p_sub = p_pts[:idx+1]
            
            # Draw cyan curve
            cv2.polylines(frame, [c_sub], isClosed=False, color=cyan, thickness=20, lineType=cv2.LINE_AA)
            cv2.circle(frame, tuple(c_sub[0]), 10, cyan, -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(c_sub[-1]), 10, cyan, -1, cv2.LINE_AA)
            
            # Draw purple curve
            cv2.polylines(frame, [p_sub], isClosed=False, color=purple, thickness=20, lineType=cv2.LINE_AA)
            cv2.circle(frame, tuple(p_sub[0]), 10, purple, -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(p_sub[-1]), 10, purple, -1, cv2.LINE_AA)
            
        # Paste original shapes on top to avoid curves crossing them
        frame[mask] = orig_img[mask]
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    main()
