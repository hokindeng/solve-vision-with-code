import cv2
import numpy as np
import imageio

def main():
    img = cv2.imread('/app/first_frame.png')
    pink_mask = cv2.inRange(img, np.array([247,85,168]), np.array([247,85,168]))
    
    # Extract medium shape from center (438, 512)
    patch = pink_mask[512-40:512+40, 438-40:438+40]
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(patch, kernel, iterations=1)
    boundary = patch - eroded
    
    boundary_pts = np.argwhere(boundary > 0)
    angles = np.arctan2(boundary_pts[:, 0] - 40, boundary_pts[:, 1] - 40)
    mapped_angles = (angles + np.pi/2 + 2*np.pi) % (2 * np.pi)
    sorted_boundary = boundary_pts[np.argsort(mapped_angles)]
    
    interior_pts = np.argwhere(eroded > 0)
    sort_keys = np.lexsort((interior_pts[:, 1], -interior_pts[:, 0]))
    sorted_interior = interior_pts[sort_keys]
    
    frames = []
    num_frames = 60
    color = np.array([247, 85, 168], dtype=np.uint8)
    
    out_cx, out_cy = 886, 512
    out_patch_top = out_cy - 40
    out_patch_left = out_cx - 40
    
    for i in range(num_frames):
        frame = img.copy()
        
        # Boundary animation
        if i >= 5:
            progress = min((i - 5) / 31.0, 1.0)
            num_b = int(progress * len(sorted_boundary))
            for b in range(num_b):
                py, px = sorted_boundary[b]
                frame[out_patch_top + py, out_patch_left + px] = color
                
        # Interior animation
        if i >= 37:
            progress = min((i - 37) / 12.0, 1.0)
            num_i = int(progress * len(sorted_interior))
            for b in range(num_i):
                py, px = sorted_interior[b]
                frame[out_patch_top + py, out_patch_left + px] = color
                
            # ensure boundary is fully drawn
            for b in range(len(sorted_boundary)):
                py, px = sorted_boundary[b]
                frame[out_patch_top + py, out_patch_left + px] = color
                
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, format='FFMPEG', codec='libx264', pixelformat='yuv420p')
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == '__main__':
    main()
