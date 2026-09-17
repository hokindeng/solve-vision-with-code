img = Image.open('first_frame.png').convert('RGB')
base = np.array(img)
# Each row shifts the three symbols one position left:
# square, plus, pentagon. Copy the pentagon's pixels.
source = base[420:590, 85:255]
mask = ((source[:,:,0] > 200) & (source[:,:,1] > 60)
        & (source[:,:,1] < 200) & (source[:,:,2] < 60))
sy, sx = np.where(mask); colors = source[sy, sx]
ty, tx = sy + 761, sx + 767
angles = np.arctan2(tx - 852, -(ty - 852)) % (2*np.pi)
question = np.zeros(base.shape[:2], dtype=bool)  # "?"
question[790:915, 810:895] = np.any(
    base[790:915, 810:895] != 255, axis=2)
for i in range(35):
    frame = base.copy()
    fade = min(i / 8, 1)
    q = base[question].astype(float)*(1-fade) + 255*fade
    frame[question] = np.rint(q).astype(np.uint8)
    progress = np.clip((i-8)/25, 0, 1)
    if progress > 0:
        visible = angles <= progress*2*np.pi
        frame[ty[visible],tx[visible]] = colors[visible]
    assert (frame[:683] == base[:683]).all()
    assert (frame[683:,:683] == base[683:,:683]).all()
    p.stdin.write(frame.tobytes())  # -> video.mp4
