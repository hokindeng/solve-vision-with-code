import numpy as np
import cv2
from PIL import Image
import os
import subprocess
import tempfile

def main():
    os.makedirs('/app/output', exist_ok=True)
    
    # 3D coordinates of the 6 blocks derived from first_frame.png
    blocks = [(0,0,0), (0,0,-1), (0,0,-2), (0,0,-3), (1,0,-3), (1,1,-3)]
    
    # Pre-calculated orthographic projection vectors and offsets
    vx = np.array([-77.5, 33.5])
    vy = np.array([64.5, 39.5])
    vz = np.array([0, -89.0])
    origin = np.array([524.0, 350.0])
    
    # Warped camera view direction for perfect depth sorting
    cx = 64.5 * -89.0
    cy = 0 - (-77.5 * -89.0)
    cz = -77.5 * 39.5 - 64.5 * 33.5

    def render(angle_deg=0):
        out = np.full((1024, 1024, 3), 255, dtype=np.uint8)
        theta = angle_deg * np.pi / 180
        
        def depth(b):
            x = b[0] + 0.5
            y = b[1] + 0.5
            z = b[2] + 0.5
            nx = x * np.cos(theta) - y * np.sin(theta)
            ny = x * np.sin(theta) + y * np.cos(theta)
            nz = z
            return -(cx * nx + cy * ny + cz * nz)
            
        sorted_blocks = sorted(blocks, key=depth, reverse=True)
        
        for b in sorted_blocks:
            pts = []
            for dx in [0, 1]:
                for dy in [0, 1]:
                    for dz in [0, 1]:
                        nx = (b[0]+dx)*np.cos(theta) - (b[1]+dy)*np.sin(theta)
                        ny = (b[0]+dx)*np.sin(theta) + (b[1]+dy)*np.cos(theta)
                        nz = b[2]+dz
                        
                        px = nx * vx[0] + ny * vy[0] + nz * vz[0] + origin[0]
                        py = nx * vx[1] + ny * vy[1] + nz * vz[1] + origin[1]
                        pts.append(np.array([px, py]))
            
            faces = [
                ([0, 1, 3, 2], [123, 101, 123]), # -x
                ([4, 5, 7, 6], [123, 101, 123]), # +x
                ([0, 1, 5, 4], [123, 101, 123]), # -y
                ([2, 3, 7, 6], [123, 101, 123]), # +y
                ([0, 2, 6, 4], [123, 101, 123]), # -z
                ([1, 3, 7, 5], [206, 168, 206]), # +z
            ]
            normals = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]
            
            for face_verts, (nx, ny, nz) in zip(faces, normals):
                f = face_verts
                if list([nx,ny,nz]) == [-1,0,0]: f = [0, 2, 3, 1]
                elif list([nx,ny,nz]) == [1,0,0]: f = [4, 5, 7, 6]
                elif list([nx,ny,nz]) == [0,-1,0]: f = [0, 1, 5, 4]
                elif list([nx,ny,nz]) == [0,1,0]: f = [2, 6, 7, 3]
                elif list([nx,ny,nz]) == [0,0,-1]: f = [0, 4, 6, 2]
                elif list([nx,ny,nz]) == [0,0,1]: f = [1, 3, 7, 5]
                
                p2d = [pts[i] for i in f]
                area = (p2d[1][0]-p2d[0][0])*(p2d[2][1]-p2d[1][1]) - (p2d[1][1]-p2d[0][1])*(p2d[2][0]-p2d[1][0])
                if area > 0:
                    c = (206, 168, 206) if nz == 1 else (123, 101, 123)
                    poly = np.array(p2d, dtype=np.int32)
                    cv2.fillPoly(out, [poly], c)
                    cv2.polylines(out, [poly], True, (0,0,0), 1)
        return out

    # Generate 21 frames for a 180 degree rotation. 
    angles = np.linspace(0, -180, 21)
    first_frame = np.array(Image.open('/app/first_frame.png').convert('RGB'))
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, angle in enumerate(angles):
            # Frame 0 is exactly the given initial frame
            if i == 0:
                frame = first_frame
            else:
                frame = render(angle)
            
            Image.fromarray(frame).save(os.path.join(tmpdir, f"frame_{i:04d}.png"))
            
        cmd = [
            'ffmpeg', '-y', '-framerate', '16',
            '-i', os.path.join(tmpdir, 'frame_%04d.png'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '/app/output/video.mp4'
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == '__main__':
    main()
