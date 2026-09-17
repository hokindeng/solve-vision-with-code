import cv2
import numpy as np
import imageio

def generate_video():
    img = cv2.imread('/app/first_frame.png')
    
    # Pre-calculated parameters for each object (identified by its unique color)
    # The parameters were found by template matching the rotated solid object's outline
    # against the dashed target outlines (which have color [90, 90, 90]).
    colors = [0, 3, 4, 5, 6]
    angles = {0: 219.8, 3: 186.0, 4: 117.6, 5: 40.4, 6: 34.6}
    dxs = {0: 515, 3: 475, 4: 483, 5: 500, 6: 482}

    bg = img.copy()
    objects_data = []

    for c in colors:
        # Find the mask of the specific color
        color_mask = cv2.imread(f'/app/color_{c}.png', cv2.IMREAD_GRAYSCALE) > 0
        if not np.any(color_mask):
            continue
            
        y, x = np.where(color_mask)
        ymin, ymax, xmin, xmax = y.min(), y.max(), x.min(), x.max()
        
        # The object also contains color 2 [120, 120, 120] as an internal pattern/border.
        # So we extract the full object by finding the connected component of non-background,
        # non-target colors that overlaps with this unique color.
        obj_mask = ~(np.all(img == [255, 255, 255], axis=-1) | np.all(img == [90, 90, 90], axis=-1))
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(obj_mask.astype(np.uint8), connectivity=8)
        
        # Find which connected component corresponds to this object
        cy, cx = (ymin+ymax)//2, (xmin+xmax)//2
        label_id = labels[cy, cx]
        single_obj_mask = (labels == label_id).astype(np.uint8)
        
        # Remove the object from the background image (replace with white)
        bg[single_obj_mask == 1] = [255, 255, 255]
        
        # Calculate the center of mass for rotation
        M_obj = cv2.moments(single_obj_mask)
        ocx = M_obj['m10'] / M_obj['m00']
        ocy = M_obj['m01'] / M_obj['m00']
        
        # Extract the object's original pixels
        obj_pixels = np.zeros_like(img)
        obj_pixels[single_obj_mask == 1] = img[single_obj_mask == 1]
        
        objects_data.append({
            'mask': single_obj_mask,
            'pixels': obj_pixels,
            'ocx': ocx,
            'ocy': ocy,
            'target_angle': angles[c],
            'target_dx': dxs[c]
        })

    frames = []
    n_frames = 48

    for t in range(n_frames):
        progress = t / (n_frames - 1)
        frame = bg.copy()
        
        for obj in objects_data:
            angle = progress * obj['target_angle']
            dx = progress * obj['target_dx']
            
            # Rotation matrix around the center of mass, plus horizontal translation
            M = cv2.getRotationMatrix2D((obj['ocx'], obj['ocy']), angle, 1.0)
            M[0, 2] += dx
            
            # Warp the mask and the pixels
            rotated_mask = cv2.warpAffine(obj['mask'], M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)
            rotated_pixels = cv2.warpAffine(obj['pixels'], M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)
            
            # Overlay the object onto the frame
            idx = rotated_mask > 0
            frame[idx] = rotated_pixels[idx]
            
        # Convert BGR to RGB for saving with imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    # Save the video
    imageio.mimwrite('/app/output/video.mp4', frames, fps=16, codec='libx264', pixelformat='yuv420p')

if __name__ == '__main__':
    generate_video()
