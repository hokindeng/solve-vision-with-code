import cv2
import numpy as np
import os
import subprocess

def main():
    img_path = '/app/first_frame.png'
    out_video_path = '/app/output/video.mp4'
    frames_dir = '/tmp/frames'
    
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Could not read /app/first_frame.png")
        
    # Extracted coordinates of dots 1 to 7
    points = [
        (888, 154),
        (424, 574),
        (171, 390),
        (783, 648),
        (653, 818),
        (630, 226),
        (315, 185)
    ]

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_video_path), exist_ok=True)

    # Mask for all pixels that are NOT pure white (the background)
    # This allows us to put the dots back on top of the drawn lines.
    not_white = np.any(img != [255, 255, 255], axis=-1)

    frames_per_line = 15
    total_lines = len(points) - 1

    frame_idx = 0

    def save_frame(idx, image):
        out = image.copy()
        # Restore non-white pixels (the dots, numbers, anti-aliasing)
        out[not_white] = img[not_white]
        cv2.imwrite(f'{frames_dir}/frame_{idx:04d}.png', out)

    # Frame 0: Start frame
    save_frame(frame_idx, img)
    frame_idx += 1

    R = 48 # Radius of dots to avoid drawing line inside the white interior of the circle
    
    # Canvas holds all completed lines
    canvas = img.copy()

    for i in range(total_lines):
        A = np.array(points[i], dtype=float)
        B = np.array(points[i+1], dtype=float)
        
        d = np.linalg.norm(B - A)
        u = (B - A) / d
        
        # Start and end points strictly outside the circle radius
        p1 = A + u * R
        p2 = B - u * R
        
        for step in range(1, frames_per_line + 1):
            progress = step / frames_per_line
            p_curr = p1 + (p2 - p1) * progress
            
            temp_canvas = canvas.copy()
            start_pt = tuple(np.round(p1).astype(int))
            end_pt = tuple(np.round(p_curr).astype(int))
            
            cv2.line(temp_canvas, start_pt, end_pt, (0, 0, 255), 5, cv2.LINE_AA)
            
            save_frame(frame_idx, temp_canvas)
            frame_idx += 1
            
        cv2.line(canvas, tuple(np.round(p1).astype(int)), tuple(np.round(p2).astype(int)), (0, 0, 255), 5, cv2.LINE_AA)

    # Hold the final frame to reach exactly 100 frames
    while frame_idx < 100:
        save_frame(frame_idx, canvas)
        frame_idx += 1

    # Run ffmpeg to encode
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-framerate', '16',
        '-i', f'{frames_dir}/frame_%04d.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        out_video_path
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    
if __name__ == "__main__":
    main()
