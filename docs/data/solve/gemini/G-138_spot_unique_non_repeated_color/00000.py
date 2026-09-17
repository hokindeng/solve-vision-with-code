import cv2
import numpy as np
import imageio

def get_unique_color_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    colors = []
    for cnt in contours:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_val = cv2.mean(img, mask=mask)[:3]
        colors.append(tuple(int(x) for x in mean_val))
    
    def color_dist(c1, c2):
        return sum(abs(a - b) for a, b in zip(c1, c2))
    
    unique_idx = -1
    for i, c1 in enumerate(colors):
        is_unique = True
        for j, c2 in enumerate(colors):
            if i != j and color_dist(c1, c2) < 20: 
                is_unique = False
                break
        if is_unique:
            unique_idx = i
            break
            
    return contours[unique_idx]

def main():
    img = cv2.imread('/app/first_frame.png')
    cnt = get_unique_color_contour(img)
    
    num_frames = 21
    out_path = '/app/output/video.mp4'
    writer = imageio.get_writer(out_path, fps=16, codec='libx264', pixelformat='yuv420p', quality=9)
    
    for i in range(num_frames):
        frame = img.copy()
        
        fraction = i / (num_frames - 1)
        num_points = int(fraction * len(cnt))
        
        if num_points > 0:
            if i == num_frames - 1:
                cv2.polylines(frame, [cnt], isClosed=True, color=(0, 0, 0), thickness=4, lineType=cv2.LINE_AA)
            else:
                pts = cnt[:num_points]
                cv2.polylines(frame, [pts], isClosed=False, color=(0, 0, 0), thickness=4, lineType=cv2.LINE_AA)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == '__main__':
    main()
