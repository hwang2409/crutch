#!/usr/bin/env python3
from rational import Rational
from polynomial_parser import parse_polynomial
from cas import CAS

def main():
	cas = CAS()
	p = parse_polynomial("(x+y)^3 - 5/2 z + 2xy")
	print(p)
	print(p.eval({'x': Rational(2,1),'y':Rational(1,1),'z':Rational(2,1)}))
	print(cas.differentiate("3x^4 + 2x^2", 'x'))
	print(cas.integrate("12x^3 + 4x", 'x'))

if __name__ == '__main__':
	main()
