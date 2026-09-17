import cv2
import numpy as np
import imageio
import os

def main():
    img = cv2.imread('/app/first_frame.png')

    # 1. Create clean background
    clean_bg = img.copy()
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]

    # Blue and Dark Blue
    agent_mask = ((b == 255) & (g == 0) & (r == 0)) | ((b == 139) & (g == 0) & (r == 0))
    clean_bg[agent_mask] = [0, 128, 0] # Replace with Start node green

    # 2. Extract agent pixels relative to start center
    agent_pixels = []
    center_x, center_y = 724, 348
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if agent_mask[y, x]:
                agent_pixels.append((x - center_x, y - center_y, img[y, x].tolist()))

    # 3. Define path coordinates
    # Start: (724, 348)
    # N1: (586, 642)
    # End: (276, 792)
    path = [
        (724, 348),
        (586, 642),
        (276, 792)
    ]

    # Calculate segment distances
    d1 = np.hypot(path[1][0] - path[0][0], path[1][1] - path[0][1])
    d2 = np.hypot(path[2][0] - path[1][0], path[2][1] - path[1][1])
    total_d = d1 + d2

    total_frames = 30
    frames = []

    for i in range(total_frames):
        # Linearly interpolate based on distance
        progress = i / (total_frames - 1)
        dist_covered = progress * total_d
        
        if dist_covered <= d1:
            if d1 == 0: t = 0
            else: t = dist_covered / d1
            x = path[0][0] + t * (path[1][0] - path[0][0])
            y = path[0][1] + t * (path[1][1] - path[0][1])
        else:
            if d2 == 0: t = 0
            else: t = (dist_covered - d1) / d2
            t = min(1.0, max(0.0, t))
            x = path[1][0] + t * (path[2][0] - path[1][0])
            y = path[1][1] + t * (path[2][1] - path[1][1])
            
        # Draw agent on clean background
        frame = clean_bg.copy()
        for dx, dy, color in agent_pixels:
            nx, ny = int(round(x)) + dx, int(round(y)) + dy
            if 0 <= nx < 1024 and 0 <= ny < 1024:
                frame[ny, nx] = color
                
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    # 4. Save video
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', macro_block_size=None, ffmpeg_params=['-pix_fmt', 'yuv420p'])
    for f in frames:
        writer.append_data(f)
    writer.close()
    print("Video generated successfully.")

if __name__ == '__main__':
    main()
