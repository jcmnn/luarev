def fnv1a(s):
    b = 0xf42e1c3e
    for c in s:
        b = ((b ^ c) * 0x1000193) & 0xFFFFFFFF
    return b

with open("names.txt") as f:
    with open("codes.txt", "w") as c:
        for line in f:
            line = line.rstrip().lstrip().encode("utf-8")
            c.write(line.decode("utf-8") + ":" + hex(fnv1a(line)) + "\n")
