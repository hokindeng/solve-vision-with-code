import cv2
import numpy as np
import imageio
import os

def find_intersection(img):
    colors, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
    line_colors = []
    for c, count in zip(colors, counts):
        if count > 100 and not np.all(c == 255):
            line_colors.append(c)

    masks = [cv2.inRange(img, c, c) for c in line_colors]

    def fit_line(mask):
        y, x = np.where(mask > 0)
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return m, c

    m1, c1 = fit_line(masks[0])
    m2, c2 = fit_line(masks[1])

    ix = (c2 - c1) / (m1 - m2)
    iy = m1 * ix + c1

    return int(round(ix)), int(round(iy))

def main():
    img = cv2.imread('/app/first_frame.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    ix, iy = find_intersection(img)
    print(f"Intersection at: ({ix}, {iy})")
    
    frames = []
    total_frames = 30
    
    radius = 40
    thickness = 5
    color = (255, 0, 0) # Red in RGB
    
    for i in range(total_frames):
        frame = img_rgb.copy()
        
        if i < 5:
            # Initial pause
            pass
        elif i <= 25:
            # Draw progressively
            progress = (i - 5) / 20.0
            angle = int(progress * 360)
            if angle > 0:
                cv2.ellipse(frame, (ix, iy), (radius, radius), 0, 0, angle, color, thickness, lineType=cv2.LINE_AA)
        else:
            # Final pause
            cv2.ellipse(frame, (ix, iy), (radius, radius), 0, 0, 360, color, thickness, lineType=cv2.LINE_AA)
            
        frames.append(frame)
        
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=1)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == '__main__':
    main()
