import cv2
import numpy as np
import imageio
import os

def main():
    # Read the first frame
    img_path = '/app/first_frame.png'
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read {img_path}")
    
    # Convert to RGB for imageio
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Find the most frequent color to assume as background
    # Downsample for speed
    small = cv2.resize(img, (100, 100))
    colors, counts = np.unique(small.reshape(-1, 3), axis=0, return_counts=True)
    bg_color = colors[np.argmax(counts)]

    # Simple color distance to find shapes
    diff = np.abs(img.astype(np.int32) - bg_color.astype(np.int32)).sum(axis=2)
    mask = (diff > 20).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shapes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 5 and h > 5:
            shapes.append({'c': c, 'x': x})
            
    if not shapes:
        raise ValueError("No shapes found")
        
    # Sort to find the leftmost shape
    shapes.sort(key=lambda s: s['x'])
    leftmost_c = shapes[0]['c']
    
    # Get the minimum enclosing circle to draw around it
    (cx, cy), radius = cv2.minEnclosingCircle(leftmost_c)
    center = (int(cx), int(cy))
    # Add some padding so the red circle surrounds the shape completely
    radius = int(radius) + 15  

    output_dir = '/app/output'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'video.mp4')
    
    writer = imageio.get_writer(
        output_path,
        fps=16,
        codec='libx264',
        format='FFMPEG',
        macro_block_size=None,
        pixelformat='yuv420p'
    )

    num_frames = 48
    for i in range(num_frames):
        frame = img_rgb.copy()
        
        # Calculate angle for the arc
        # i=0 -> angle=0
        # i=47 -> angle=360
        end_angle = (i / (num_frames - 1)) * 360
        
        if end_angle > 0:
            # Draw an arc growing from the top (-90 degrees) clockwise
            cv2.ellipse(frame, center, (radius, radius), -90, 0, end_angle, (255, 0, 0), 8, cv2.LINE_AA)
            
        writer.append_data(frame)

    writer.close()

if __name__ == '__main__':
    main()
