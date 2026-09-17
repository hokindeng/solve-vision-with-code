import cv2
import numpy as np
import imageio
import os

def main():
    img = cv2.imread('/app/first_frame.png')
    if img is None:
        raise FileNotFoundError("Could not read /app/first_frame.png")
    
    # We only modify the D patch (bottom-right quadrant)
    # The quadrant is at y=512..1024, x=512..1024
    D_orig = img[512:, 512:].copy()
    
    # The shape to copy is in the C patch (bottom-left quadrant)
    C = img[512:, :512]
    
    # Extract the green shape from C
    maskC = (C == [53, 153, 70]).all(axis=-1).astype(np.uint8) * 255
    
    # The transformation (as observed from A -> B) is an erosion with 3x3 kernel
    kernel = np.ones((3, 3), np.uint8)
    maskC_eroded = cv2.erode(maskC, kernel)
    
    # The boundary that is removed during the transformation
    diff_mask = cv2.bitwise_xor(maskC, maskC_eroded)
    
    # We shift the masks by 192 pixels to the right to place them correctly in D
    # center of C is 160.5. Center of D is 352.5. (352.5 - 160.5 = 192.0)
    M = np.float32([[1, 0, 192], [0, 1, 0]])
    shifted_eroded = cv2.warpAffine(maskC_eroded, M, (512, 512))
    shifted_diff = cv2.warpAffine(diff_mask, M, (512, 512))
    
    # Prepare D_clean which has the '?' removed (everything right of x=100 is replaced with white)
    D_clean = D_orig.copy()
    D_clean[:, 100:] = 255
    
    # Green color for the shape
    green = np.array([53, 153, 70], dtype=np.float32)
    
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    
    num_frames = 60
    for i in range(num_frames):
        # Calculate animation opacities
        if i <= 10:
            # Phase 1: Crossfade from '?' to full C shape
            alpha_shape = i / 10.0
            alpha_q = 1.0 - alpha_shape
            alpha_bound = 1.0
        else:
            # Phase 2: Fade out the boundary (erosion)
            alpha_shape = 1.0
            alpha_q = 0.0
            alpha_bound = 1.0 - (i - 10) / 49.0
            
        D_curr = D_orig.astype(np.float32).copy()
        
        # 1. Apply the background crossfade for x > 100
        D_curr[:, 100:] = D_orig[:, 100:].astype(np.float32) * alpha_q + D_clean[:, 100:].astype(np.float32) * (1.0 - alpha_q)
        
        # 2. Add the eroded shape
        mask_eroded_3d = (shifted_eroded[:, :, None] > 0).astype(np.float32)
        D_curr = D_curr * (1.0 - mask_eroded_3d * alpha_shape) + green * (mask_eroded_3d * alpha_shape)
        
        # 3. Add the boundary
        mask_diff_3d = (shifted_diff[:, :, None] > 0).astype(np.float32)
        D_curr = D_curr * (1.0 - mask_diff_3d * alpha_shape * alpha_bound) + green * (mask_diff_3d * alpha_shape * alpha_bound)
        
        # Reconstruct the full frame
        frame = img.copy()
        frame[512:, 512:] = np.clip(np.round(D_curr), 0, 255).astype(np.uint8)
        
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        writer.append_data(frame_rgb)
        
    writer.close()
    print('Video generated successfully.')

if __name__ == '__main__':
    main()
