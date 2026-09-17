img = Image.open('first_frame.png').convert('RGB')
original = np.array(img)
orange = np.all(original == (255,165,0), axis=2) # agent
sy, sx = np.where(orange); dx, dy = sx - 255, sy - 459
board = original.copy(); board[orange] = (0, 255, 0)
# Each leg has Manhattan-minimal length: 5, 8, 8, 8.
corners = [(2,4), (2,8), (1,8), (3,8), (3,2),
           (8,2), (8,5), (8,4), (1,4)]
route = [corners[0]]
for target in corners[1:]:
    x, y = route[-1]
    while (x, y) != target:
        x += int(np.sign(target[0]-x))
        y += int(np.sign(target[1]-y))
        route.append((x, y))
centers = np.array([(51+102*x, 51+102*y)
                    for x, y in route])
for frame_idx in range(107):   # hold, travel, hold
    t = np.clip((frame_idx-3)/99, 0, 1) * (len(route)-1)
    i = min(int(t), len(route)-2)
    p = np.rint(centers[i]*(1-(t-i))
                + centers[i+1]*(t-i)).astype(int)
    frame = board.copy()
    frame[p[1]+dy, p[0]+dx] = (255, 165, 0)
    proc.stdin.write(frame.tobytes())  # -> video.mp4
