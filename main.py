#!/usr/bin/env python3
from rational import Rational
from cas import CAS

def main():
	cas = CAS()
	print(cas.differentiate("3x^4 + 2x^2", 'x'))
	print(cas.integrate("12x^3 + 4x", 'x'))
	# General expression parsing fallback (non-polynomial): exponent non-integer, division by variable
	print(cas.eval("x^0.5 + 1/x", {'x': 4}))
	# Basic functions via general expression evaluator
	print(cas.eval("sin(0) + cos(0)", {}))
	# EDAG-based differentiation on composite functions (chain/product/quotient/power rules)
	print(cas.differentiate("sin(x^2)", 'x'))
	print(cas.differentiate("exp(x^2 + 3x)", 'x'))
	print(cas.differentiate("log(exp(x))", 'x'))
	# Inverse trig derivatives
	print(cas.differentiate("asin(x)", 'x'))
	print(cas.differentiate("acos(x)", 'x'))
	print(cas.differentiate("atan(x)", 'x'))
	print(cas.differentiate("tan(sin(x))", 'x'))
	print(cas.eval("log(e^2) + ln(e^2)", {}))
	print(cas.eval("-x", {"x": 2}))
	print(cas.differentiate("x^2", "x"))

	r = cas.parse("x^3 + exp(x)")
	print(r.eval({'x': 2}))

if __name__ == '__main__':
	main()
