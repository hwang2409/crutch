#!/usr/bin/env python3
from cas import CAS
from interval import Interval
import math

def main():
	cas = CAS()
	
	f2 = cas.parse("x^2 - 4*x + 3")
	roots = f2.find_all_roots('x', Interval.reals())
	print(f"Roots of x^2 - 4x + 3: {roots}")




if __name__ == '__main__':
	main()
