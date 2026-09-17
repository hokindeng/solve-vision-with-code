import os
import cv2
import numpy as np
from PIL import Image
import subprocess
import shutil

def main():
    os.makedirs('/tmp/frames', exist_ok=True)
    
    # Load first frame
    bg_img = Image.open('/app/first_frame.png').convert('RGBA')
    bg_np = np.array(bg_img)
    
    # 5 dominos to fall
    start_xs = [138, 214, 269, 332, 392]
    
    # Erase dominos 0-4 to create clean background
    for start_x in start_xs:
        # Erase domino body (make white)
        bg_np[583:721, start_x:start_x+46] = [255, 255, 255, 255]
        # Erase bottom outline and restore ground line
        bg_np[721:723, start_x:start_x+46] = [160, 82, 45, 255]
        
    bg_clean = Image.fromarray(bg_np)
    
    # Create domino layers
    original_np = np.array(bg_img)
    domino_layers = []
    for start_x in start_xs:
        layer = np.zeros((1024, 1024, 4), dtype=np.uint8)
        layer[583:723, start_x:start_x+46] = original_np[583:723, start_x:start_x+46]
        domino_layers.append(Image.fromarray(layer))
        
    # Animation parameters
    T_start = [0, 8.1, 12.6, 18.7, 24.2]
    hit_angles = [12.3, 3.7, 6.9, 5.7, 0]
    final_angles = [56.0, 61.0, 66.0, 71.0, 90.0]
    D = 22.0
    
    def free_angle(t):
        if t <= 0: return 0.0
        return min(90.0, 90.0 * (t / D)**2)
        
    angles = [[0]*5 for _ in range(54)]
    
    for t in range(54):
        a4 = free_angle(t - T_start[4])
        angles[t][4] = min(a4, final_angles[4])
        
        for i in range(3, -1, -1):
            a_free = free_angle(t - T_start[i])
            a_next = angles[t][i+1]
            prog = a_next / final_angles[i+1] if final_angles[i+1] > 0 else 0
            allowed_diff = (1 - prog) * hit_angles[i] + prog * (final_angles[i] - final_angles[i+1])
            a_constrained = a_next + allowed_diff
            angles[t][i] = max(0.0, min(a_free, a_constrained))
            
    # Draw frames
    for t in range(54):
        frame = bg_clean.copy()
        for i in range(4, -1, -1):
            angle = angles[t][i]
            if angle > 0:
                pivot = (start_xs[i] + 46, 723)
                # Rotate clockwise is negative angle in PIL
                rotated_layer = domino_layers[i].rotate(-angle, center=pivot, resample=Image.BICUBIC)
                frame.alpha_composite(rotated_layer)
            else:
                frame.alpha_composite(domino_layers[i])
                
        # Convert to RGB before saving
        frame = frame.convert('RGB')
        frame.save(f'/tmp/frames/frame_{t:04d}.png')
        
    # Generate video
    if os.path.exists('/app/output/video.mp4'):
        os.remove('/app/output/video.mp4')
        
    cmd = [
        'ffmpeg', '-y', '-framerate', '16', '-i', '/tmp/frames/frame_%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '/app/output/video.mp4'
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Video generated successfully.")

if __name__ == '__main__':
    main()
