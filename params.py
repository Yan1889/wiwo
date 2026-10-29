#!/bin/python3
import sys
arr = sys.argv[1].split("-")
res = 0
for i in range(1, len(arr)):
    res += int(arr[i - 1]) * int(arr[i])
print(res)
