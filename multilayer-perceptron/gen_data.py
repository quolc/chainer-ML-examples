# (x, y) in [0, 1] * [0, 1]
# x > y なら0, x < y なら1
import sys
import random
import math

n = int(sys.argv[1])
print(n)

for i in range(0, n):
    x = random.random()*2 - 1.0
    y = random.random()*2 - 1.0
#    theta = math.atan2(x, y)
#    r = math.sqrt(x*x + y*y)
#    t = 0 if math.sin(r*math.pi*2 - theta) > 0 else 1
    t = 0 if (x < 0) ^ (y < 0) else 1
    print("{0}\t{1}\t{2}".format(x, y, t))
