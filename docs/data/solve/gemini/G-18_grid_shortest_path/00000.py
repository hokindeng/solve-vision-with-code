import cv2
import numpy as np
import imageio

def get_agent_sprite(img):
    green_mask = np.all(img == [0, 255, 0], axis=-1)
    coords = np.argwhere(green_mask)
    min_y, max_y = coords[:, 0].min(), coords[:, 0].max()
    min_x, max_x = coords[:, 1].min(), coords[:, 1].max()
    sprite_mask = green_mask[min_y:max_y+1, min_x:max_x+1]
    return sprite_mask

def main():
    img = cv2.imread('first_frame.png')
    sprite_mask = get_agent_sprite(img)
    
    # Create clean background
    bg = img.copy()
    green_pixels = np.all(bg == [0, 255, 0], axis=-1)
    bg[green_pixels] = [0, 165, 255] # Fill start cell with orange
    
    # Path of cells
    # (6, 1) -> (8, 7)
    # Right 6 times, Down 2 times
    path = []
    for c in range(1, 8):
        path.append((6, c))
    for r in range(7, 9):
        path.append((r, 7))
    
    num_frames = 60
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', pixelformat='yuv420p')
    
    for i in range(num_frames):
        if i == 0:
            frame = img.copy()
        else:
            frame = bg.copy()
            
            # Calculate position
            total_segments = len(path) - 1
            progress = i / (num_frames - 1) * total_segments
            seg = int(progress)
            t = progress - seg
            
            if seg >= total_segments:
                seg = total_segments - 1
                t = 1.0
                
            r1, c1 = path[seg]
            r2, c2 = path[seg+1]
            
            y1 = 2 + r1*102 + 15
            x1 = 2 + c1*102 + 15
            y2 = 2 + r2*102 + 15
            x2 = 2 + c2*102 + 15
            
            curr_y = int(round(y1 + (y2 - y1) * t))
            curr_x = int(round(x1 + (x2 - x1) * t))
            
            # Paste sprite
            # sprite_mask is 69x69
            h, w = sprite_mask.shape
            
            # Paste only where mask is True
            # We assume it doesn't go out of bounds since it's far from the edges of the 1024x1024 image
            frame_roi = frame[curr_y:curr_y+h, curr_x:curr_x+w]
            frame_roi[sprite_mask] = [0, 255, 0]
            
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    main()
