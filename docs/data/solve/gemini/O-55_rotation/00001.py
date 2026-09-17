import numpy as np
import cv2
import os
import subprocess

def get_proj(az_deg):
    az = np.deg2rad(az_deg)
    el = np.deg2rad(36)
    S = 138.0
    dx_u = S * np.cos(az)
    dx_v = S * np.sin(az) * np.sin(el)
    dy_u = S * -np.sin(az)
    dy_v = S * np.cos(az) * np.sin(el)
    dz_u = 0.0
    dz_v = -S * np.cos(el)
    return np.array([[dx_u, dy_u, dz_u], [dx_v, dy_v, dz_v]])

def get_depth_vec(az_deg):
    az = np.deg2rad(az_deg)
    el = np.deg2rad(36)
    return np.array([np.sin(az)*np.cos(el), np.cos(az)*np.cos(el), np.sin(el)])

C_TOP = (225, 168, 197)
C_RIGHT = (180, 135, 157)
C_LEFT = (135, 101, 118)

faces_info = [
    ([4, 5, 6, 7], np.array([0, 0, 1]), C_TOP),
    ([3, 2, 1, 0], np.array([0, 0, -1]), C_TOP),
    ([1, 5, 6, 2], np.array([1, 0, 0]), C_RIGHT),
    ([0, 4, 7, 3], np.array([-1, 0, 0]), C_RIGHT),
    ([3, 2, 6, 7], np.array([0, 1, 0]), C_LEFT),
    ([0, 1, 5, 4], np.array([0, -1, 0]), C_LEFT),
]

def render_az(az_deg):
    P = get_proj(az_deg)
    d_vec = get_depth_vec(az_deg)
    img = np.full((1024, 1024, 3), 255, dtype=np.uint8)
    blocks = [(-1, -1, 2), (-1, -1, 1), (-1, 0, 2), (-1, 0, 1), (-1, 0, 0), (0, 0, 0)]
    
    # Sort blocks by depth (further blocks first)
    blocks = sorted(blocks, key=lambda b: (b[0]+0.5)*d_vec[0] + (b[1]+0.5)*d_vec[1] + (b[2]+0.5)*d_vec[2])
    
    for b in blocks:
        x, y, z = b
        verts = np.array([
            [x, y, z], [x+1, y, z], [x+1, y+1, z], [x, y+1, z],
            [x, y, z+1], [x+1, y, z+1], [x+1, y+1, z+1], [x, y+1, z+1]
        ])
        pts = (verts @ P.T + np.array([513, 680])).astype(np.int32)
        
        # Sort faces within block by depth (further faces first, or just draw visible ones)
        # Actually, if we only draw faces pointing towards camera, order doesn't matter much for convex block.
        # But to be safe, sort faces by depth of their centroid!
        visible_faces = []
        for idxs, normal, color in faces_info:
            if np.dot(normal, d_vec) > -1e-5: # visible (allow slightly facing away to avoid edge gaps)
                centroid = np.mean(verts[idxs], axis=0)
                depth = np.dot(centroid, d_vec)
                visible_faces.append((depth, idxs, color))
                
        visible_faces = sorted(visible_faces, key=lambda f: f[0])
        
        for depth, idxs, color in visible_faces:
            face_pts = pts[idxs]
            cv2.fillPoly(img, [face_pts], color)
            cv2.polylines(img, [face_pts], True, (0, 0, 0), 1, cv2.LINE_8)
    return img

def main():
    os.makedirs('/app/output', exist_ok=True)
    frames_dir = '/app/frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    azimuths = np.linspace(10, 190, 21)
    
    for i, az in enumerate(azimuths):
        if i == 0:
            # Use exact first frame
            img = cv2.imread('/app/first_frame.png')
        else:
            img = render_az(az)
        cv2.imwrite(f'{frames_dir}/frame_{i:03d}.png', img)
        
    subprocess.run([
        'ffmpeg', '-y', '-framerate', '16', '-i', f'{frames_dir}/frame_%03d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ], check=True)

if __name__ == '__main__':
    main()
