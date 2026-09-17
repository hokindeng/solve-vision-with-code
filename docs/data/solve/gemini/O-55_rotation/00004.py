import numpy as np
import cv2
import subprocess
import os

def create_video():
    # Blocks
    blocks = [
        (0, 0, 0),
        (1, 0, 0),
        (2, 0, 0),
        (2, 1, 0),
        (2, 1, 1),
        (2, 2, 0)
    ]

    colors = {
        'top': (159, 197, 225),   # OpenCV is BGR, these were extracted from image
        'left': (127, 157, 180),
        'right': (95, 118, 135),
    }

    S = 137.5
    phi = np.deg2rad(28)

    # Calculate offset using theta=20
    theta_0 = np.deg2rad(20)
    def project_offset(x, y, z):
        u = S * (x * np.cos(theta_0) + y * np.sin(theta_0))
        v = S * (x * np.sin(theta_0) * np.sin(phi) - y * np.cos(theta_0) * np.sin(phi) - z * np.cos(phi))
        return np.array([u, v])

    corners_3d = [(-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, -0.5, -0.5), (-0.5, -0.5, -0.5)]
    pts = [project_offset(x, y, z) for x, y, z in corners_3d]
    min_u = min(p[0] for p in pts)
    min_v = min(p[1] for p in pts)
    offset = np.array([248.5 - min_u, 571.0 - min_v])

    faces_def = {
        '+x': [(0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5), (0.5, -0.5, 0.5)],
        '-x': [(-0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), (-0.5, -0.5, 0.5)],
        '+y': [(-0.5, 0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)],
        '-y': [(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (-0.5, -0.5, 0.5)],
        '+z': [(-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)],
        '-z': [(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5)]
    }
    normals = {'+x': [1,0,0], '-x': [-1,0,0], '+y': [0,1,0], '-y': [0,-1,0], '+z': [0,0,1], '-z': [0,0,-1]}

    os.makedirs('/app/output', exist_ok=True)
    frames_dir = '/app/frames'
    os.makedirs(frames_dir, exist_ok=True)

    first_frame = cv2.imread('/app/first_frame.png')
    
    num_frames = 21
    
    for i in range(num_frames):
        if i == 0:
            cv2.imwrite(f'{frames_dir}/frame_{i:04d}.png', first_frame)
            continue
            
        theta_deg = 20 + 180 * (i / (num_frames - 1))
        theta = np.deg2rad(theta_deg)
        
        img = np.full((1024, 1024, 3), 255, dtype=np.uint8)
        
        def project(x, y, z):
            u = S * (x * np.cos(theta) + y * np.sin(theta))
            v = S * (x * np.sin(theta) * np.sin(phi) - y * np.cos(theta) * np.sin(phi) - z * np.cos(phi))
            return np.array([u + offset[0], v + offset[1]])

        W_dir = np.array([np.sin(theta)*np.cos(phi), -np.cos(theta)*np.cos(phi), np.sin(phi)])
        blocks_sorted = sorted(blocks, key=lambda b: np.dot(np.array(b), W_dir))

        for b in blocks_sorted:
            cx, cy, cz = b
            
            # depth sort faces of this block
            visible_faces = []
            for fname, fcorners in faces_def.items():
                n = normals[fname]
                if np.dot(n, W_dir) <= 1e-5:
                    continue
                # compute center of face
                center = np.mean(fcorners, axis=0) + np.array([cx, cy, cz])
                depth = np.dot(center, W_dir)
                visible_faces.append((depth, fname, fcorners, n))
            
            visible_faces.sort(key=lambda x: x[0])
            
            for depth, fname, fcorners, n in visible_faces:
                if fname == '+z':
                    color = colors['top']
                else:
                    nu = n[0] * np.cos(theta) + n[1] * np.sin(theta)
                    color = colors['left'] if nu < 0 else colors['right']
                    
                pts_2d = []
                for x, y, z in fcorners:
                    pts_2d.append(project(cx+x, cy+y, cz+z))
                
                pts_2d = np.round(pts_2d).astype(np.int32)
                cv2.fillPoly(img, [pts_2d], color, lineType=cv2.LINE_8)
                cv2.polylines(img, [pts_2d], True, (0,0,0), 1, cv2.LINE_8)
                
        cv2.imwrite(f'{frames_dir}/frame_{i:04d}.png', img)

    cmd = [
        'ffmpeg', '-y', '-framerate', '16', '-i', f'{frames_dir}/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ]
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    create_video()
