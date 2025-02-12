p1 = [0, 0]
p2 = [20, 0]
p3 = [40, 5]
p4 = [40, -5]
p5 = [60, 40]
p6 = [60, 45]
nets = [[p1, p2], [p3, p5], [p4, p6]]
count = 0
for i, net_i in enumerate(nets[:-1]):
    for j, net_j in enumerate(nets[i + 1 :]):
        a, b = net_i
        c, d = net_j
        if not (
            max(a[0], b[0]) < min(c[0], d[0])
            or max(c[0], d[0]) < min(a[0], b[0])
            or max(a[1], b[1]) < min(c[1], d[1])
            or max(c[1], d[1]) < min(a[1], b[1])
        ):
            count += 1
print(count)
