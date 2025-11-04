#!/usr/bin/env python3
from cas import CAS
from interval import Interval
import math

def main():
	cas = CAS()
	
	# Test monotonicity checks
	print("=== Monotonicity Tests ===")
	
	# x^2: decreasing on (-∞, 0), increasing on (0, ∞)
	f_x2 = cas.parse("x^2")
	print(f"x^2 is strictly increasing on (0, ∞): {f_x2.is_strictly_increasing('x', Interval.open(0, math.inf))}")
	print(f"x^2 is strictly decreasing on (-∞, 0): {f_x2.is_strictly_decreasing('x', Interval.open(-math.inf, 0))}")
	
	# x^3: strictly increasing everywhere
	f_x3 = cas.parse("x^3")
	print(f"x^3 is strictly increasing on ℝ: {f_x3.is_strictly_increasing('x')}")
	
	# -x: strictly decreasing everywhere
	f_negx = cas.parse("-x")
	print(f"-x is strictly decreasing on ℝ: {f_negx.is_strictly_decreasing('x')}")
	
	# x^2 - 4x + 3: has minimum at x=2
	f_quad = cas.parse("x^2 - 4*x + 3")
	print(f"x^2 - 4x + 3 is strictly decreasing on (-∞, 2): {f_quad.is_strictly_decreasing('x', Interval.open(-math.inf, 2))}")
	print(f"x^2 - 4x + 3 is strictly increasing on (2, ∞): {f_quad.is_strictly_increasing('x', Interval.open(2, math.inf))}")
	
	# exp(x): strictly increasing everywhere
	f_exp = cas.parse("exp(x)")
	print(f"exp(x) is strictly increasing on ℝ: {f_exp.is_strictly_increasing('x')}")
	
	# -exp(x): strictly decreasing everywhere
	f_negexp = cas.parse("-exp(x)")
	print(f"-exp(x) is strictly decreasing on ℝ: {f_negexp.is_strictly_decreasing('x')}")
	
	# x^2 on [0, ∞): monotone increasing (allows equality at x=0)
	print(f"x^2 is monotone increasing on [0, ∞): {f_x2.is_monotone_increasing('x', Interval.closed(0, math.inf))}")
	
	print("\n=== Root Finding Tests ===")
	f1 = cas.parse("exp(x) - e")
	roots = f1.find_all_roots('x', Interval.reals())
	print(f"Roots of exp(x) - e: {roots}")

	f2 = cas.parse("x^2 - 4*x + 3")
	roots = f2.find_all_roots('x', Interval.reals())
	print(f"Roots of x^2 - 4x + 3: {roots}")




if __name__ == '__main__':
	main()
