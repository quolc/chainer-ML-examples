# (x, y) in [0, 1] * [0, 1]
# x > y なら0, x < y なら1
import sys
import random

n = int(sys.argv[1])
print(n)

for i in range(0, n):
    x = random.random()
    y = random.random()
    t = 0 if x > y else 1
    print("{0}\t{1}\t{2}".format(x, y, t))
