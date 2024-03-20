with open('time.txt') as topo_file:
    y = 0
    for line in topo_file:
        x = float(line)
        y += x

    print(y/822)