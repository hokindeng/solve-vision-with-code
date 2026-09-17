import cv2
import numpy as np
import math
import os
import imageio

def get_proj_matrix(az_deg, el_deg):
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    f = np.array([-math.cos(az)*math.cos(el), -math.sin(az)*math.cos(el), -math.sin(el)])
    up = np.array([0., 0., 1.])
    s = np.cross(f, up)
    if np.linalg.norm(s) > 1e-6:
        s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    return s, -u, f

def render_frame(az_deg, el_deg):
    blocks = [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 2, 1), (0, 3, 1), (-1, 3, 1)]
    s, u, f_vec = get_proj_matrix(az_deg, el_deg)
    
    face_defs = []
    for (bx, by, bz) in blocks:
        corners = [
            (bx, by, bz), (bx+1, by, bz), (bx+1, by+1, bz), (bx, by+1, bz),
            (bx, by, bz+1), (bx+1, by, bz+1), (bx+1, by+1, bz+1), (bx, by+1, bz+1)
        ]
        defs = [
            ((150, 150, 206), [corners[4], corners[5], corners[6], corners[7]], np.array([bx+0.5, by+0.5, bz+1]), np.array([0,0,1])),
            ((150, 150, 206), [corners[0], corners[3], corners[2], corners[1]], np.array([bx+0.5, by+0.5, bz]), np.array([0,0,-1])),
            ((120, 120, 165), [corners[1], corners[2], corners[6], corners[5]], np.array([bx+1, by+0.5, bz+0.5]), np.array([1,0,0])),
            ((120, 120, 165), [corners[0], corners[4], corners[7], corners[3]], np.array([bx, by+0.5, bz+0.5]), np.array([-1,0,0])),
            ((90, 90, 123), [corners[3], corners[7], corners[6], corners[2]], np.array([bx+0.5, by+1, bz+0.5]), np.array([0,1,0])),
            ((90, 90, 123), [corners[0], corners[1], corners[5], corners[4]], np.array([bx+0.5, by, bz+0.5]), np.array([0,-1,0]))
        ]
        for color, pts, center, normal in defs:
            if np.dot(normal, f_vec) < -1e-6:
                depth = np.dot(center, f_vec)
                face_defs.append((depth, color, pts))
                
    face_defs.sort(key=lambda x: x[0], reverse=True)
    
    origin_2d = (441.2, 523.6)
    scale = 103.467
    
    img = np.full((1024, 1024, 3), 255, dtype=np.uint8)
    for depth, color, pts in face_defs:
        pts_2d = []
        for pt in pts:
            x = origin_2d[0] + scale * (pt[0]*s[0] + pt[1]*s[1] + pt[2]*s[2])
            y = origin_2d[1] + scale * (pt[0]*u[0] + pt[1]*u[1] + pt[2]*u[2])
            pts_2d.append([int(round(x)), int(round(y))])
        pts_2d = np.array(pts_2d, dtype=np.int32)
        cv2.fillPoly(img, [pts_2d], color, lineType=cv2.LINE_8)
        cv2.polylines(img, [pts_2d], True, (0, 0, 0), 1, lineType=cv2.LINE_8)
        
    return img

def main():
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')

    for i in range(21):
        az = 70 + (250 - 70) * i / 20.0
        
        if i == 0:
            frame_bgr = cv2.imread('/app/first_frame.png')
        else:
            frame_bgr = render_frame(az, 25)
            
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)

    writer.close()

if __name__ == '__main__':
    main()
