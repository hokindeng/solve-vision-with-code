import cv2
import numpy as np
import PIL.Image
import imageio
import os

def main():
    # 1. Load first frame
    first_frame = np.array(PIL.Image.open('/app/first_frame.png').convert('RGB'))
    
    # 2. Extract q_image and arrow_image
    # Source region for the Solid Arrow (C)
    src_y1, src_y2 = 670, 860
    src_x1, src_x2 = 60, 250
    arrow_image = first_frame[src_y1:src_y2, src_x1:src_x2].copy()
    
    # Destination region for ? and the final animated arrow
    dst_y1, dst_y2 = 670, 860
    dst_x1, dst_x2 = 764, 954
    q_image = first_frame[dst_y1:dst_y2, dst_x1:dst_x2].copy()
    
    # 3. Compute contour and parameterize the solid arrow outline
    # Mask for the Solid Arrow (using a threshold)
    gray_arrow = cv2.cvtColor(arrow_image, cv2.COLOR_RGB2GRAY)
    mask = (gray_arrow < 128).astype(np.uint8) * 255
    
    # Distance transform to find the ridge/skeleton
    dist_img = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, ridge = cv2.threshold(dist_img, 2.0, 255, cv2.THRESH_BINARY)
    
    # Find contour of the ridge
    contours, _ = cv2.findContours(ridge.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # The longest contour is our path
    cnt = sorted(contours, key=cv2.contourArea)[-1].squeeze()
    
    # Compute cumulative distance along the contour
    diffs = np.linalg.norm(np.diff(cnt, axis=0), axis=1)
    dists = np.concatenate(([0], np.cumsum(diffs)))
    total_len = dists[-1] + np.linalg.norm(cnt[-1] - cnt[0])
    
    # 60 dashes as inferred from B
    num_dashes = 60
    period = total_len / num_dashes
    pixel_phase = 1.0 / period
    
    # Map each pixel in the arrow mask to a distance s along the contour
    y_coords, x_coords = np.where(mask > 0)
    s_vals = np.zeros(len(y_coords))
    for i, (y, x) in enumerate(zip(y_coords, x_coords)):
        # Find closest point on contour
        d = np.sum((cnt - [x, y])**2, axis=1)
        idx = np.argmin(d)
        s_vals[i] = dists[idx]
        
    # Phase in [0, 1] for the dash period
    phase = (s_vals % period) / period
    # Distance to the center of the dash period
    dist_to_center = np.abs(phase - 0.5)
    
    start_ratio = 1.0 + 2 * pixel_phase
    end_ratio = 0.35  # empirically matched with B
    
    # 4. Generate frames
    frames = []
    num_frames = 60
    crossfade_frames = 15
    dash_frames = num_frames - 1 - crossfade_frames # 44
    
    for f in range(num_frames):
        frame_img = first_frame.copy()
        
        if f == 0:
            # Frame 0 is untouched (shows the question mark)
            pass
        elif f <= crossfade_frames:
            # Frames 1 to 15: Crossfade ? to Solid Arrow
            alpha = f / float(crossfade_frames)
            # Blend: q_image fades out (to white), arrow_image fades in (from white)
            blended = (q_image.astype(float) * (1.0 - alpha) + arrow_image.astype(float) * alpha).astype(np.uint8)
            frame_img[dst_y1:dst_y2, dst_x1:dst_x2] = blended
        else:
            # Frames 16 to 59: Animate dashes
            progress = (f - crossfade_frames) / float(dash_frames)
            current_dash_ratio = start_ratio - (start_ratio - end_ratio) * progress
            
            edge_phase = current_dash_ratio / 2.0
            # Continuous mask values for anti-aliased gaps
            mask_values = np.clip((edge_phase - dist_to_center) / pixel_phase + 0.5, 0.0, 1.0)
            
            # Construct 2D mask patch (default 1.0 for white background)
            current_mask_2d = np.ones((dst_y2 - dst_y1, dst_x2 - dst_x1), dtype=float)
            current_mask_2d[y_coords, x_coords] = mask_values
            
            # Apply mask: where mask is 0.0, fade arrow to white (255)
            blended = (arrow_image.astype(float) * current_mask_2d[:,:,None] + 
                       255.0 * (1.0 - current_mask_2d[:,:,None])).astype(np.uint8)
                       
            frame_img[dst_y1:dst_y2, dst_x1:dst_x2] = blended
            
        frames.append(frame_img)
        
    # 5. Save video
    os.makedirs('/app/output', exist_ok=True)
    out_path = '/app/output/video.mp4'
    
    # Use imageio to write H.264 mp4
    writer = imageio.get_writer(out_path, fps=16, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    for fr in frames:
        writer.append_data(fr)
    writer.close()
    
    print(f"Video saved to {out_path}")

if __name__ == '__main__':
    main()
