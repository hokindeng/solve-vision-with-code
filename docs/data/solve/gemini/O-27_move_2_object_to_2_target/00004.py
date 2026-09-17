import cv2
import numpy as np
import imageio
import os

def main():
    img = cv2.imread('/app/first_frame.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Define masks for the objects
    mask1 = ((img_rgb[:,:,0] == 135) & (img_rgb[:,:,1] == 206) & (img_rgb[:,:,2] == 235)) | \
            ((img_rgb[:,:,0] == 105) & (img_rgb[:,:,1] == 176) & (img_rgb[:,:,2] == 205)) | \
            ((img_rgb[:,:,0] == 195) & (img_rgb[:,:,1] == 255) & (img_rgb[:,:,2] == 255))

    mask2 = ((img_rgb[:,:,0] == 64) & (img_rgb[:,:,1] == 224) & (img_rgb[:,:,2] == 208)) | \
            ((img_rgb[:,:,0] == 34) & (img_rgb[:,:,1] == 194) & (img_rgb[:,:,2] == 178))

    # Create background by replacing objects with background color
    bg = img.copy()
    bg[mask1] = [220, 220, 220]
    bg[mask2] = [220, 220, 220]

    y1, x1 = np.where(mask1)
    y2, x2 = np.where(mask2)
    obj1_pixels = img[mask1]
    obj2_pixels = img[mask2]

    # Target translations (shortest path perfectly aligned to center)
    target_dx1 = -104
    target_dy1 = 350
    target_dx2 = -323
    target_dy2 = -26

    num_frames = 35
    fps = 16

    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer(
        '/app/output/video.mp4', 
        fps=fps, 
        codec='libx264',
        pixelformat='yuv420p',
        macro_block_size=None,
        quality=9
    )

    for i in range(num_frames):
        t = i / (num_frames - 1)
        dx1 = int(round(target_dx1 * t))
        dy1 = int(round(target_dy1 * t))
        dx2 = int(round(target_dx2 * t))
        dy2 = int(round(target_dy2 * t))

        frame = bg.copy()
        
        # We can just apply the shift directly to the coordinates
        nx1 = x1 + dx1
        ny1 = y1 + dy1
        frame[ny1, nx1] = obj1_pixels

        nx2 = x2 + dx2
        ny2 = y2 + dy2
        frame[ny2, nx2] = obj2_pixels

        # Convert back to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)

    writer.close()

if __name__ == '__main__':
    main()
