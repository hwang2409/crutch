#!/usr/bin/env python3
from cas import CAS
from interval import Interval
import math

def main():
    cas = CAS()
    f1 = cas.parse("x^2 - 2x + 1")
    sols = f1.solve('x')
    for x in sols:
        print(x)

if __name__ == "__main__":
    main()
