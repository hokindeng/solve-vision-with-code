import cv2
import numpy as np
import os
import subprocess

def get_squares(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    colors = np.unique(img.reshape(-1, 3), axis=0)
    bg_color = img[50, 50]
    squares = []
    for c in colors:
        if np.array_equal(c, bg_color):
            continue
        mask = cv2.inRange(img, c, c)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            squares.append({
                'color': c,
                'bbox': (x, y, w, h),
                'area': w * h
            })
    squares.sort(key=lambda s: s['area'], reverse=True)
    return [s['bbox'] for s in squares]

def draw_rect(img, x1, y1, x2, y2, color, thickness):
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

def interpolate(val1, val2, alpha):
    return val1 + (val2 - val1) * alpha

def generate_video():
    img_orig = cv2.imread('/app/first_frame.png')
    squares = get_squares(img_orig)
    
    if len(squares) < 3:
        squares = [
            (125, 125, 775, 775),
            (252, 252, 521, 521),
            (385, 385, 255, 255)
        ]
        
    sq1 = squares[0]
    sq2 = squares[1]
    sq3 = squares[2]
    
    os.makedirs('/app/output', exist_ok=True)
    
    red = (0, 0, 255)
    blue = (255, 0, 0)
    thickness = 6
    offset = 3
    
    def get_rect(sq):
        x, y, w, h = sq
        return (x - offset, y - offset, x + w + offset, y + h + offset)
        
    r1 = get_rect(sq1)
    r2 = get_rect(sq2)
    r3 = get_rect(sq3)

    writer = cv2.VideoWriter('/app/output/temp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 16, (1024, 1024))
    
    for f in range(85):
        img = img_orig.copy()
        
        if 0 <= f < 10:
            draw_rect(img, r1[0], r1[1], r1[2], r1[3], red, thickness)
        elif 10 <= f < 20:
            alpha = (f - 10) / 10.0
            x1 = interpolate(r1[0], r2[0], alpha)
            y1 = interpolate(r1[1], r2[1], alpha)
            x2 = interpolate(r1[2], r2[2], alpha)
            y2 = interpolate(r1[3], r2[3], alpha)
            draw_rect(img, x1, y1, x2, y2, red, thickness)
        elif 20 <= f < 30:
            draw_rect(img, r2[0], r2[1], r2[2], r2[3], red, thickness)
        elif 30 <= f < 40:
            alpha = (f - 30) / 10.0
            x1 = interpolate(r2[0], r3[0], alpha)
            y1 = interpolate(r2[1], r3[1], alpha)
            x2 = interpolate(r2[2], r3[2], alpha)
            y2 = interpolate(r2[3], r3[3], alpha)
            draw_rect(img, x1, y1, x2, y2, red, thickness)
        elif 40 <= f < 46:
            draw_rect(img, r3[0], r3[1], r3[2], r3[3], red, thickness)
        elif 46 <= f < 50:
            pass
        elif f >= 50:
            bx1, by1, bx2, by2 = r3
            
            if f > 57:
                cv2.line(img, (int(bx1), int(by1)), (int(bx2), int(by1)), blue, thickness)
            elif 50 <= f <= 57:
                alpha = (f - 50 + 1) / 8.0
                curr_x = int(interpolate(bx1, bx2, alpha))
                cv2.line(img, (int(bx1), int(by1)), (curr_x, int(by1)), blue, thickness)
                
            if f > 65:
                cv2.line(img, (int(bx2), int(by1)), (int(bx2), int(by2)), blue, thickness)
            elif 58 <= f <= 65:
                alpha = (f - 58 + 1) / 8.0
                curr_y = int(interpolate(by1, by2, alpha))
                cv2.line(img, (int(bx2), int(by1)), (int(bx2), curr_y), blue, thickness)
                
            if f > 73:
                cv2.line(img, (int(bx2), int(by2)), (int(bx1), int(by2)), blue, thickness)
            elif 66 <= f <= 73:
                alpha = (f - 66 + 1) / 8.0
                curr_x = int(interpolate(bx2, bx1, alpha))
                cv2.line(img, (int(bx2), int(by2)), (curr_x, int(by2)), blue, thickness)
                
            if f > 81:
                cv2.line(img, (int(bx1), int(by2)), (int(bx1), int(by1)), blue, thickness)
            elif 74 <= f <= 81:
                alpha = (f - 74 + 1) / 8.0
                curr_y = int(interpolate(by2, by1, alpha))
                cv2.line(img, (int(bx1), int(by2)), (int(bx1), curr_y), blue, thickness)
                
        writer.write(img)
        
    writer.release()
    
    subprocess.run([
        'ffmpeg', '-y', '-i', '/app/output/temp.mp4', 
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '/app/output/video.mp4'
    ], check=True)
    os.remove('/app/output/temp.mp4')

if __name__ == '__main__':
    generate_video()
