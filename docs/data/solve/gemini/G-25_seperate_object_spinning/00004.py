import cv2
import numpy as np
import imageio
import os

def solve():
    img = cv2.imread('/app/first_frame.png')
    bg_color = img[0,0]
    
    diff = cv2.absdiff(img, np.full_like(img, bg_color))
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    solids = []
    dashed = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 40 and h > 40:
            solids.append(c)
        else:
            dashed.append(c)
            
    dashed_img = np.zeros_like(thresh)
    cv2.drawContours(dashed_img, dashed, -1, 255, -1)
    dashed_img_dilated = cv2.dilate(dashed_img, np.ones((3,3), np.uint8), iterations=1)
            
    solid_data = []
    for i, c in enumerate(solids):
        M = cv2.moments(c)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        
        mask = np.zeros_like(thresh)
        cv2.drawContours(mask, [c], -1, 255, -1)
        
        def eval_angle(angle):
            M_rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            rotated_mask = cv2.warpAffine(mask, M_rot, (mask.shape[1], mask.shape[0]))
            rx, ry, rw, rh = cv2.boundingRect(rotated_mask)
            template_mask = rotated_mask[ry:ry+rh, rx:rx+rw]
            
            template_outline = np.zeros_like(template_mask)
            tc, _ = cv2.findContours(template_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(template_outline, tc, -1, 255, 2)
            
            res = cv2.matchTemplate(dashed_img_dilated, template_outline, cv2.TM_CCORR)
            
            best_score_local = -1
            best_dx_local = 0
            
            pt_y = ry
            for offset in [-1, 0, 1]:
                py = pt_y + offset
                if 0 <= py < res.shape[0]:
                    start_x = rx + 10
                    if start_x < res.shape[1]:
                        row = res[py, start_x:]
                        score = np.max(row)
                        if score > best_score_local:
                            best_score_local = score
                            best_dx_local = np.argmax(row) + start_x - rx
                            
            return best_score_local, best_dx_local

        best_score = -1
        best_angle = 0
        best_dx = 0
        
        for angle in range(0, 360, 3):
            score, dx = eval_angle(angle)
            if score > best_score:
                best_score = score
                best_angle = angle
                best_dx = dx
                
        best_score_2 = -1
        best_angle_2 = best_angle
        best_dx_2 = best_dx
        for angle in range(best_angle - 2, best_angle + 3):
            a = angle % 360
            score, dx = eval_angle(a)
            if score > best_score_2:
                best_score_2 = score
                best_angle_2 = a
                best_dx_2 = dx
                
        target_angle = best_angle_2
        delta_angle = target_angle
        if delta_angle > 180:
            delta_angle -= 360
            
        obj_colored = cv2.bitwise_and(img, img, mask=mask)
        
        solid_data.append({
            'contour': c,
            'cx': cx,
            'cy': cy,
            'mask': mask,
            'obj_colored': obj_colored,
            'delta_angle': delta_angle,
            'target_dx': best_dx_2
        })
        
    base_frame = img.copy()
    for data in solid_data:
        cv2.drawContours(base_frame, [data['contour']], -1, tuple(int(x) for x in bg_color), -1)
        
    frames = []
    num_frames = 48
    
    for f in range(num_frames):
        if f <= 23:
            rot_t = f / 23.0
            trans_t = 0.0
        else:
            rot_t = 1.0
            trans_t = (f - 23) / 24.0
            
        frame = base_frame.copy()
        
        for data in solid_data:
            angle = data['delta_angle'] * rot_t
            dx = data['target_dx'] * trans_t
            
            M_rot = cv2.getRotationMatrix2D((data['cx'], data['cy']), angle, 1.0)
            M_rot[0, 2] += dx
            
            warped_obj = cv2.warpAffine(data['obj_colored'], M_rot, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            warped_mask = cv2.warpAffine(data['mask'], M_rot, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            _, warped_mask = cv2.threshold(warped_mask, 127, 255, cv2.THRESH_BINARY)
            
            inv_mask = cv2.bitwise_not(warped_mask)
            frame = cv2.bitwise_and(frame, frame, mask=inv_mask)
            frame = cv2.add(frame, warped_obj)
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    os.makedirs('/app/output', exist_ok=True)
    writer = imageio.get_writer('/app/output/video.mp4', fps=16, codec='libx264', macro_block_size=None, pixelformat='yuv420p')
    for frm in frames:
        writer.append_data(frm)
    writer.close()

if __name__ == '__main__':
    solve()
