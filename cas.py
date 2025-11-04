from __future__ import annotations
from typing import Dict, Any, Union, Optional
from parser import parse_polynomial, parse_expression_edag
from polynomial import Polynomial
from rational import Rational
from edag import EDAG
from interval import Interval
import re
import math

NumberLike = Union[int, float, Rational]

DEFAULT_LOWER_BOUND = -10000.0
DEFAULT_UPPER_BOUND = 10000.0

class CAS:
	def __init__(self) -> None:
		pass

	def _wrap(self, obj: Any) -> 'CAS.ExprResult':
		if isinstance(obj, Polynomial):
			return CAS.ExprResult(poly=obj)
		if isinstance(obj, EDAG):
			return CAS.ExprResult(dag=obj)
		if isinstance(obj, CAS.ExprResult):
			return obj
		raise TypeError('Unsupported object for wrapping')

	def parse(self, expr: str) -> 'CAS.ExprResult':
		# Prefer EDAG for general expressions
		if re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\(", expr) or re.search(r"\d+\.\d+", expr):
			dag = parse_expression_edag(expr)
			return CAS.ExprResult(dag=dag)
		# Try polynomial first
		try:
			p = parse_polynomial(expr)
			return CAS.ExprResult(poly=p)
		except Exception:
			dag = parse_expression_edag(expr)
			return CAS.ExprResult(dag=dag)
	class ExprResult:
		def __init__(self, poly: Polynomial | None = None, dag: EDAG | None = None) -> None:
			self._poly = poly
			self._dag = dag
		def eval(self, env: Dict[str, Any] | None = None) -> Any:
			env = env or {}
			if self._poly is not None:
				env_r = {k: (v if isinstance(v, Rational) else Rational(v,1) if isinstance(v,int) else Rational(int(v*10**6),10**6)) for k,v in env.items()}
				return self._poly.eval(env_r)
			if self._dag is not None:
				return self._dag.eval(env)
			raise RuntimeError('empty expression')
		def __str__(self) -> str:
			if self._poly is not None:
				return str(self._poly)
			if self._dag is not None:
				return self._dag.to_string()
			return ''
		def _eval_float(self, var: str, x: float) -> float:
			"""Evaluate expression at x, returning float."""
			val = self.eval({var: x})
			if isinstance(val, Rational):
				return val.numerator() / val.denominator()
			return float(val)
		def find_root(self, var: str, interval: Interval, tol: float = 1e-10, max_iter: int = 100) -> Optional[float]:
			"""Find a root of f(x) = 0 in the given interval using Brent's method.
			
			Returns the root if found, None if no root exists in the interval.
			Requires the interval to bracket a root (opposite signs at endpoints).
			"""
			if interval.is_empty():
				return None
			# Get endpoints
			a = interval.a
			b = interval.b
			# Handle unbounded intervals
			if a == -math.inf:
				a = -100.0
			if b == math.inf:
				b = 100.0
			# Adjust for open endpoints
			if interval.left_open and a != -math.inf:
				a += 1e-6
			if interval.right_open and b != math.inf:
				b -= 1e-6
			if a >= b:
				return None
			# Evaluate at endpoints
			fa = self._eval_float(var, a)
			fb = self._eval_float(var, b)
			# Check if interval brackets a root
			if fa * fb > 0:
				return None  # Same sign, no root guaranteed
			# Ensure a < b and |fb| <= |fa| (b is the better guess)
			if a > b:
				a, b = b, a
				fa, fb = fb, fa
			if abs(fa) < abs(fb):
				a, b = b, a
				fa, fb = fb, fa
				# After swap, ensure a < b again
				if a > b:
					a, b = b, a
					fa, fb = fb, fa
			c = a
			fc = fa
			mflag = True
			d = a
			for _ in range(max_iter):
				if abs(fb) < tol:
					return b
				if abs(b - a) < tol:
					return b
				# Use bisection if conditions are not met
				if fa != fc and fb != fc and abs(fa - fb) > 1e-15:
					# Inverse quadratic interpolation
					s = (a * fb * fc) / ((fa - fb) * (fa - fc)) + \
						(b * fa * fc) / ((fb - fa) * (fb - fc)) + \
						(c * fa * fb) / ((fc - fa) * (fc - fb))
				elif abs(fb - fa) > 1e-15:
					# Secant method
					s = b - fb * (b - a) / (fb - fa)
				else:
					# fa == fb, use bisection
					s = (a + b) / 2.0
					mflag = True
				# Check conditions for using bisection
				condition1 = (s <= (3*a + b) / 4 or s >= b)
				condition2 = (mflag and abs(s - b) >= abs(b - c) / 2)
				condition3 = (not mflag and abs(s - b) >= abs(c - d) / 2)
				condition4 = (mflag and abs(b - c) < tol)
				condition5 = (not mflag and abs(c - d) < tol)
				if condition1 or condition2 or condition3 or condition4 or condition5:
					s = (a + b) / 2.0
					mflag = True
				else:
					mflag = False
				fs = self._eval_float(var, s)
				d = c
				c = b
				fc = fb
				if fa * fs < 0:
					b = s
					fb = fs
				else:
					a = s
					fa = fs
				# Ensure a < b and |fb| <= |fa|
				if a > b:
					a, b = b, a
					fa, fb = fb, fa
				if abs(fa) < abs(fb):
					a, b = b, a
					fa, fb = fb, fa
					if a > b:
						a, b = b, a
						fa, fb = fb, fa
			# Max iterations reached
			if abs(fb) < tol:
				return b
			return None
		def find_all_roots(self, var: str, interval: Interval, tol: float = 1e-8, max_iter: int = 100, sample_points: int = 1000) -> list[float]:
			"""Find all roots of f(x) = 0 in the given interval.
			
			Returns a list of roots found. Uses sampling to detect sign changes,
			then applies Brent's method to each bracketed interval.
			"""
			if interval.is_empty():
				return []
			roots = []
			# Get finite bounds for sampling
			a = interval.a
			b = interval.b
			if a == -math.inf:
				a = DEFAULT_LOWER_BOUND
			if b == math.inf:
				b = DEFAULT_UPPER_BOUND
			if interval.left_open and a != -math.inf:
				a += 1e-6
			if interval.right_open and b != math.inf:
				b -= 1e-6
			if a >= b:
				return []
			# Sample points to find sign changes
			x_vals = []
			f_vals = []
			for i in range(sample_points + 1):
				x = a + i * (b - a) / sample_points
				try:
					fx = self._eval_float(var, x)
					x_vals.append(x)
					f_vals.append(fx)
				except Exception:
					continue
			# Use a more lenient tolerance for detecting roots during sampling
			sample_tol = max(tol * 100, 1e-6)
			# First pass: find all sample points that are near zero
			potential_roots = set()
			for i in range(len(f_vals)):
				if abs(f_vals[i]) < sample_tol:
					potential_roots.add(i)
			# Second pass: find sign changes (these may contain roots not at sample points)
			brackets = []
			for i in range(len(f_vals) - 1):
				if f_vals[i] * f_vals[i+1] < 0:
					# Clear sign change detected - bracket for Brent's method
					brackets.append((x_vals[i], x_vals[i+1]))
				elif abs(f_vals[i]) < sample_tol or abs(f_vals[i+1]) < sample_tol:
					# One endpoint is zero or very small - need to check for sign changes
					# This could be a root, or could have a sign change nearby
					# Check if there's a sign change by looking at adjacent intervals
					has_sign_change = False
					# Check previous interval if exists
					if i > 0:
						prev_f = f_vals[i-1]
						curr_f = f_vals[i] if abs(f_vals[i]) < sample_tol else f_vals[i+1]
						if abs(f_vals[i]) < sample_tol:
							# f_vals[i] is zero, check if prev_f and f_vals[i+1] have opposite signs
							if prev_f * f_vals[i+1] < 0:
								has_sign_change = True
								brackets.append((x_vals[i-1], x_vals[i+1]))
						else:
							# f_vals[i+1] is zero, check if f_vals[i] and next_f have opposite signs
							if i + 1 < len(f_vals) - 1:
								next_f = f_vals[i+2]
								if f_vals[i] * next_f < 0:
									has_sign_change = True
									brackets.append((x_vals[i], x_vals[i+2]))
					# Also check next interval if exists
					if not has_sign_change and i + 1 < len(f_vals) - 1:
						if abs(f_vals[i+1]) < sample_tol:
							next_f = f_vals[i+2]
							if f_vals[i] * next_f < 0:
								has_sign_change = True
								brackets.append((x_vals[i], x_vals[i+2]))
					# If no sign change detected, still sample densely to find roots
					if not has_sign_change:
						# Sample the interval more densely
						interval_width = x_vals[i+1] - x_vals[i]
						# Use adaptive sampling: more samples for wider intervals
						num_samples = max(20, min(100, int(interval_width / max(1, (b - a) / sample_points) * 10)))
						prev_x = x_vals[i]
						prev_f = f_vals[i]
						for j in range(1, num_samples + 1):
							test_x = x_vals[i] + j * interval_width / (num_samples + 1)
							try:
								test_f = self._eval_float(var, test_x)
								# Check for sign change (handle zeros specially)
								sign_change = False
								if prev_f * test_f < 0:
									# Clear sign change
									sign_change = True
								elif abs(prev_f) < sample_tol and abs(test_f) > sample_tol:
									# prev is zero, test is non-zero - check if it's a crossing
									if j > 1:
										# Check the value before prev
										prev_prev_x = x_vals[i] + (j - 1) * interval_width / (num_samples + 1)
										try:
											prev_prev_f = self._eval_float(var, prev_prev_x)
											if prev_prev_f * test_f < 0:
												sign_change = True
										except Exception:
											pass
									# Also try to bracket from prev to test
									if not sign_change:
										dx = interval_width / (num_samples + 1)
										refine_a = max(a, prev_x - dx)
										refine_b = min(b, test_x + dx)
										fa_test = self._eval_float(var, refine_a)
										fb_test = self._eval_float(var, refine_b)
										if fa_test * fb_test < 0:
											brackets.append((refine_a, refine_b))
											sign_change = True
								elif abs(test_f) < sample_tol and abs(prev_f) > sample_tol:
									# test is zero, prev is non-zero - check if it's a crossing
									dx = interval_width / (num_samples + 1)
									refine_a = max(a, prev_x - dx)
									refine_b = min(b, test_x + dx)
									fa_test = self._eval_float(var, refine_a)
									fb_test = self._eval_float(var, refine_b)
									if fa_test * fb_test < 0:
										brackets.append((refine_a, refine_b))
										sign_change = True
								if sign_change:
									brackets.append((prev_x, test_x))
								prev_x = test_x
								prev_f = test_f
							except Exception:
								pass
				else:
					# Neither endpoint is near zero, but we should still check for roots
					# Sample points in the interval to detect sign changes or near-zero values
					interval_width = x_vals[i+1] - x_vals[i]
					# Use adaptive number of samples based on interval size
					# For smaller intervals, sample more densely
					base_step = (b - a) / sample_points
					if interval_width <= base_step * 10:
						# Small interval - sample more densely
						num_samples = max(10, int(interval_width / base_step) * 2)
					else:
						# Larger interval - sample less densely but still check
						num_samples = 10
					prev_x = x_vals[i]
					prev_f = f_vals[i]
					for j in range(1, num_samples + 1):
						test_x = x_vals[i] + j * interval_width / (num_samples + 1)
						try:
							test_f = self._eval_float(var, test_x)
							# Check for sign change
							if prev_f * test_f < 0:
								brackets.append((prev_x, test_x))
							# Check if this point is near zero
							if abs(test_f) < sample_tol:
								# Try to bracket it
								dx = interval_width / (num_samples + 1)
								refine_a = max(a, test_x - dx * 3)
								refine_b = min(b, test_x + dx * 3)
								fa_test = self._eval_float(var, refine_a)
								fb_test = self._eval_float(var, refine_b)
								if fa_test * fb_test < 0:
									brackets.append((refine_a, refine_b))
							prev_x = test_x
							prev_f = test_f
						except Exception:
							pass
			# Refine potential roots found at sample points
			for idx in potential_roots:
				x_sample = x_vals[idx]
				f_sample = f_vals[idx]
				# If already very close to zero, use it directly
				if abs(f_sample) < tol:
					roots.append(x_sample)
				else:
					# Try to refine using Brent's method
					# Create interval around sample point
					dx = max((b - a) / sample_points * 20, 1e-2)
					refine_a = max(a, x_sample - dx)
					refine_b = min(b, x_sample + dx)
					# Check if this interval brackets a root
					fa_check = self._eval_float(var, refine_a)
					fb_check = self._eval_float(var, refine_b)
					if fa_check * fb_check < 0:
						# Properly brackets - use Brent's method
						refine_interval = Interval.closed(refine_a, refine_b)
						refined = self.find_root(var, refine_interval, tol, max_iter)
						if refined is not None:
							roots.append(refined)
					elif abs(fa_check) < tol:
						roots.append(refine_a)
					elif abs(fb_check) < tol:
						roots.append(refine_b)
					else:
						# No sign change, but function is small - try Newton's method style refinement
						# Just use the sample point if it's close enough
						if abs(f_sample) < sample_tol:
							roots.append(x_sample)
			# Also check if endpoints are roots
			if abs(f_vals[0]) < sample_tol:
				refine_a = a
				refine_b = min(b, a + (b - a) / sample_points)
				refine_interval = Interval.closed(refine_a, refine_b)
				refined = self.find_root(var, refine_interval, tol, max_iter)
				if refined is not None:
					roots.append(refined)
			if abs(f_vals[-1]) < sample_tol and len(f_vals) > 1:
				refine_a = max(a, b - (b - a) / sample_points)
				refine_b = b
				refine_interval = Interval.closed(refine_a, refine_b)
				refined = self.find_root(var, refine_interval, tol, max_iter)
				if refined is not None:
					roots.append(refined)
			# Use Brent's method on each bracketed interval
			for a_bracket, b_bracket in brackets:
				# Verify the bracket actually has opposite signs
				fa_br = self._eval_float(var, a_bracket)
				fb_br = self._eval_float(var, b_bracket)
				if fa_br * fb_br < 0:
					sub_interval = Interval.closed(a_bracket, b_bracket)
					root = self.find_root(var, sub_interval, tol, max_iter)
					if root is not None:
						roots.append(root)
				elif abs(fa_br) < tol:
					roots.append(a_bracket)
				elif abs(fb_br) < tol:
					roots.append(b_bracket)
			# Also check intervals where function is small but no sign change detected
			# This catches cases where function touches zero without crossing
			# or where we might have missed a root due to sampling density
			for i in range(len(f_vals) - 1):
				# If function is small at both endpoints, check for root
				if abs(f_vals[i]) < sample_tol * 10 and abs(f_vals[i+1]) < sample_tol * 10:
					check_interval = Interval.closed(x_vals[i], x_vals[i+1])
					root = self.find_root(var, check_interval, tol, max_iter)
					if root is not None:
						roots.append(root)
				# Also check if function magnitude is small in the interval (might indicate nearby root)
				elif min(abs(f_vals[i]), abs(f_vals[i+1])) < sample_tol:
					# One endpoint is very small - check the interval more carefully
					# Sample at a few points in the interval
					for j in range(3):
						test_x = x_vals[i] + (j + 1) * (x_vals[i+1] - x_vals[i]) / 4
						try:
							test_f = self._eval_float(var, test_x)
							if abs(test_f) < sample_tol:
								# Found a near-zero point - try to refine
								dx = (x_vals[i+1] - x_vals[i]) / 4
								refine_a = max(a, test_x - dx)
								refine_b = min(b, test_x + dx)
								fa_test = self._eval_float(var, refine_a)
								fb_test = self._eval_float(var, refine_b)
								if fa_test * fb_test < 0:
									refine_interval = Interval.closed(refine_a, refine_b)
									refined = self.find_root(var, refine_interval, tol, max_iter)
									if refined is not None:
										roots.append(refined)
								elif abs(test_f) < tol:
									roots.append(test_x)
						except Exception:
							pass
			# Deduplicate roots that are very close (within tolerance)
			if not roots:
				return []
			roots.sort()
			unique_roots = [roots[0]]
			for r in roots[1:]:
				if abs(r - unique_roots[-1]) > tol * 10:  # More lenient deduplication
					unique_roots.append(r)
			return unique_roots
		def _get_derivative(self, var: str) -> Optional['CAS.ExprResult']:
			"""Get derivative of this expression with respect to var."""
			if self._poly is not None:
				return CAS.ExprResult(poly=self._poly.derivative(var))
			if self._dag is not None:
				return CAS.ExprResult(dag=self._dag.derivative(var))
			return None
		def is_strictly_increasing(self, var: str, interval: Interval = None, tol: float = 1e-8) -> bool:
			"""Check if function is strictly increasing on the given interval.
			
			A function is strictly increasing if f'(x) > 0 for all x in the interval.
			
			Args:
				var: Variable name
				interval: Interval to check (defaults to all reals)
				tol: Tolerance for checking derivative sign
			
			Returns:
				True if function is strictly increasing, False otherwise
			"""
			if interval is None:
				interval = Interval.reals()
			deriv = self._get_derivative(var)
			if deriv is None:
				return False
			# Find all critical points (roots of derivative)
			# Use a reasonable bounded interval for root finding if interval is unbounded
			if interval.a == -math.inf or interval.b == math.inf:
				# Use a reasonable bounded interval for checking
				search_bounds = Interval.closed(-10.0, 10.0)
				if interval.a != -math.inf:
					search_bounds = Interval.closed(interval.a, 10.0)
				if interval.b != math.inf:
					search_bounds = Interval.closed(-10.0, interval.b)
				if interval.a != -math.inf and interval.b != math.inf:
					search_bounds = interval
				all_critical = deriv.find_all_roots(var, search_bounds, tol=tol)
				# Filter critical points to those actually in the interval
				critical_points = []
				for cp in all_critical:
					# Check if cp is in the interval
					in_interval = True
					if interval.a != -math.inf:
						if interval.left_open:
							if cp <= interval.a:
								in_interval = False
						else:
							if cp < interval.a:
								in_interval = False
					if interval.b != math.inf:
						if interval.right_open:
							if cp >= interval.b:
								in_interval = False
						else:
							if cp > interval.b:
								in_interval = False
					if in_interval:
						critical_points.append(cp)
			else:
				critical_points = deriv.find_all_roots(var, interval, tol=tol)
			# Get interval bounds for testing (use reasonable bounds if unbounded)
			# Use more conservative bounds to avoid overflow/extreme values
			if interval.a == -math.inf:
				a = -10.0
			else:
				a = interval.a
				if interval.left_open:
					a += 1e-6
			if interval.b == math.inf:
				b = 10.0
			else:
				b = interval.b
				if interval.right_open:
					b -= 1e-6
			# If no critical points, just check at a few sample points
			if not critical_points:
				# Check at a few points in the interval, avoiding boundaries
				# Use points that are slightly away from boundaries to avoid precision issues
				boundary_margin = max(1e-3, (b - a) * 0.01)  # 1% margin or 1e-3, whichever is larger
				effective_a = a + boundary_margin
				effective_b = b - boundary_margin
				if effective_a < effective_b:
					test_vals = [effective_a + (effective_b - effective_a) * i / 4 for i in range(5)]
				else:
					# Very small interval, test at midpoint
					test_vals = [(a + b) / 2]
				for test_x in test_vals:
					try:
						deriv_val = deriv._eval_float(var, test_x)
						if deriv_val <= tol:
							return False
					except (OverflowError, ValueError):
						# If overflow, try a point closer to zero
						continue
				return True
			# Check derivative sign in each subinterval
			test_points = [a]
			test_points.extend(critical_points)
			test_points.append(b)
			test_points = sorted(set(test_points))
			# Check derivative at midpoints of intervals
			for i in range(len(test_points) - 1):
				mid = (test_points[i] + test_points[i+1]) / 2
				try:
					deriv_val = deriv._eval_float(var, mid)
					if deriv_val <= tol:  # Not strictly positive
						return False
				except (OverflowError, ValueError):
					# If overflow, skip this interval (might be at extreme values)
					continue
			return True
		def is_strictly_decreasing(self, var: str, interval: Interval = None, tol: float = 1e-8) -> bool:
			"""Check if function is strictly decreasing on the given interval.
			
			A function is strictly decreasing if f'(x) < 0 for all x in the interval.
			
			Args:
				var: Variable name
				interval: Interval to check (defaults to all reals)
				tol: Tolerance for checking derivative sign
			
			Returns:
				True if function is strictly decreasing, False otherwise
			"""
			if interval is None:
				interval = Interval.reals()
			deriv = self._get_derivative(var)
			if deriv is None:
				return False
			# Find all critical points (roots of derivative)
			# Use a reasonable bounded interval for root finding if interval is unbounded
			if interval.a == -math.inf or interval.b == math.inf:
				search_bounds = Interval.closed(-10.0, 10.0)
				if interval.a != -math.inf:
					search_bounds = Interval.closed(interval.a, 10.0)
				if interval.b != math.inf:
					search_bounds = Interval.closed(-10.0, interval.b)
				if interval.a != -math.inf and interval.b != math.inf:
					search_bounds = interval
				all_critical = deriv.find_all_roots(var, search_bounds, tol=tol)
				# Filter critical points to those actually in the interval
				critical_points = []
				for cp in all_critical:
					# Check if cp is in the interval
					in_interval = True
					if interval.a != -math.inf:
						if interval.left_open:
							if cp <= interval.a:
								in_interval = False
						else:
							if cp < interval.a:
								in_interval = False
					if interval.b != math.inf:
						if interval.right_open:
							if cp >= interval.b:
								in_interval = False
						else:
							if cp > interval.b:
								in_interval = False
					if in_interval:
						critical_points.append(cp)
			else:
				critical_points = deriv.find_all_roots(var, interval, tol=tol)
			# Get interval bounds for testing (use reasonable bounds if unbounded)
			# Use more conservative bounds to avoid overflow/extreme values
			if interval.a == -math.inf:
				a = -10.0
			else:
				a = interval.a
				if interval.left_open:
					a += 1e-6
			if interval.b == math.inf:
				b = 10.0
			else:
				b = interval.b
				if interval.right_open:
					b -= 1e-6
			# If no critical points, just check at a few sample points
			if not critical_points:
				boundary_margin = max(1e-3, (b - a) * 0.01)
				effective_a = a + boundary_margin
				effective_b = b - boundary_margin
				if effective_a < effective_b:
					test_vals = [effective_a + (effective_b - effective_a) * i / 4 for i in range(5)]
				else:
					test_vals = [(a + b) / 2]
				for test_x in test_vals:
					try:
						deriv_val = deriv._eval_float(var, test_x)
						if deriv_val >= -tol:
							return False
					except (OverflowError, ValueError):
						continue
				return True
			# Check derivative sign in each subinterval
			test_points = [a]
			test_points.extend(critical_points)
			test_points.append(b)
			test_points = sorted(set(test_points))
			# Check derivative at midpoints of intervals
			for i in range(len(test_points) - 1):
				mid = (test_points[i] + test_points[i+1]) / 2
				try:
					deriv_val = deriv._eval_float(var, mid)
					if deriv_val >= -tol:  # Not strictly negative
						return False
				except (OverflowError, ValueError):
					continue
			return True
		def is_monotone_increasing(self, var: str, interval: Interval = None, tol: float = 1e-8) -> bool:
			"""Check if function is monotone increasing (non-decreasing) on the given interval.
			
			A function is monotone increasing if f'(x) >= 0 for all x in the interval.
			
			Args:
				var: Variable name
				interval: Interval to check (defaults to all reals)
				tol: Tolerance for checking derivative sign
			
			Returns:
				True if function is monotone increasing, False otherwise
			"""
			if interval is None:
				interval = Interval.reals()
			deriv = self._get_derivative(var)
			if deriv is None:
				return False
			# Find all critical points (roots of derivative)
			if interval.a == -math.inf or interval.b == math.inf:
				search_bounds = Interval.closed(-10.0, 10.0)
				if interval.a != -math.inf:
					search_bounds = Interval.closed(interval.a, 10.0)
				if interval.b != math.inf:
					search_bounds = Interval.closed(-10.0, interval.b)
				if interval.a != -math.inf and interval.b != math.inf:
					search_bounds = interval
				critical_points = deriv.find_all_roots(var, search_bounds, tol=tol)
			else:
				critical_points = deriv.find_all_roots(var, interval, tol=tol)
			# Get interval bounds for testing (use reasonable bounds if unbounded)
			# Use more conservative bounds to avoid overflow/extreme values
			if interval.a == -math.inf:
				a = -10.0
			else:
				a = interval.a
				if interval.left_open:
					a += 1e-6
			if interval.b == math.inf:
				b = 10.0
			else:
				b = interval.b
				if interval.right_open:
					b -= 1e-6
			# If no critical points, just check at a few sample points
			if not critical_points:
				boundary_margin = max(1e-3, (b - a) * 0.01)
				effective_a = a + boundary_margin
				effective_b = b - boundary_margin
				if effective_a < effective_b:
					test_vals = [effective_a + (effective_b - effective_a) * i / 4 for i in range(5)]
				else:
					test_vals = [(a + b) / 2]
				for test_x in test_vals:
					try:
						deriv_val = deriv._eval_float(var, test_x)
						if deriv_val < -tol:
							return False
					except (OverflowError, ValueError):
						continue
				return True
			# Check derivative sign in each subinterval
			test_points = [a]
			test_points.extend(critical_points)
			test_points.append(b)
			test_points = sorted(set(test_points))
			# Check derivative at midpoints of intervals
			for i in range(len(test_points) - 1):
				mid = (test_points[i] + test_points[i+1]) / 2
				try:
					deriv_val = deriv._eval_float(var, mid)
					if deriv_val < -tol:  # Negative (not non-negative)
						return False
				except (OverflowError, ValueError):
					continue
			return True
		def is_monotone_decreasing(self, var: str, interval: Interval = None, tol: float = 1e-8) -> bool:
			"""Check if function is monotone decreasing (non-increasing) on the given interval.
			
			A function is monotone decreasing if f'(x) <= 0 for all x in the interval.
			
			Args:
				var: Variable name
				interval: Interval to check (defaults to all reals)
				tol: Tolerance for checking derivative sign
			
			Returns:
				True if function is monotone decreasing, False otherwise
			"""
			if interval is None:
				interval = Interval.reals()
			deriv = self._get_derivative(var)
			if deriv is None:
				return False
			# Find all critical points (roots of derivative)
			if interval.a == -math.inf or interval.b == math.inf:
				search_bounds = Interval.closed(-10.0, 10.0)
				if interval.a != -math.inf:
					search_bounds = Interval.closed(interval.a, 10.0)
				if interval.b != math.inf:
					search_bounds = Interval.closed(-10.0, interval.b)
				if interval.a != -math.inf and interval.b != math.inf:
					search_bounds = interval
				critical_points = deriv.find_all_roots(var, search_bounds, tol=tol)
			else:
				critical_points = deriv.find_all_roots(var, interval, tol=tol)
			# Get interval bounds for testing (use reasonable bounds if unbounded)
			# Use more conservative bounds to avoid overflow/extreme values
			if interval.a == -math.inf:
				a = -10.0
			else:
				a = interval.a
				if interval.left_open:
					a += 1e-6
			if interval.b == math.inf:
				b = 10.0
			else:
				b = interval.b
				if interval.right_open:
					b -= 1e-6
			# If no critical points, just check at a few sample points
			if not critical_points:
				boundary_margin = max(1e-3, (b - a) * 0.01)
				effective_a = a + boundary_margin
				effective_b = b - boundary_margin
				if effective_a < effective_b:
					test_vals = [effective_a + (effective_b - effective_a) * i / 4 for i in range(5)]
				else:
					test_vals = [(a + b) / 2]
				for test_x in test_vals:
					try:
						deriv_val = deriv._eval_float(var, test_x)
						if deriv_val > tol:
							return False
					except (OverflowError, ValueError):
						continue
				return True
			# Check derivative sign in each subinterval
			test_points = [a]
			test_points.extend(critical_points)
			test_points.append(b)
			test_points = sorted(set(test_points))
			# Check derivative at midpoints of intervals
			for i in range(len(test_points) - 1):
				mid = (test_points[i] + test_points[i+1]) / 2
				try:
					deriv_val = deriv._eval_float(var, mid)
					if deriv_val > tol:  # Positive (not non-positive)
						return False
				except (OverflowError, ValueError):
					continue
			return True
		def find_multivariate_root(self, vars: list[str], initial_guess: Dict[str, float], tol: float = 1e-8, max_iter: int = 100) -> Optional[Dict[str, float]]:
			"""Find a root of f(x₁, x₂, ..., xₙ) = 0 using multivariate Newton's method.
			
			Args:
				vars: List of variable names
				initial_guess: Dictionary mapping variable names to initial values
				tol: Tolerance for convergence
				max_iter: Maximum iterations
			
			Returns:
				Dictionary mapping variables to root values, or None if not converged
			"""
			if len(vars) == 0:
				return None
			if len(vars) == 1:
				# Fall back to univariate method
				var = vars[0]
				guess = initial_guess.get(var, 0.0)
				interval = Interval.closed(guess - 1.0, guess + 1.0)
				root = self.find_root(var, interval, tol, max_iter)
				if root is not None:
					return {var: root}
				return None
			# Multivariate Newton's method
			x = [initial_guess.get(v, 0.0) for v in vars]
			for iteration in range(max_iter):
				# Evaluate function at current point
				env = {vars[i]: x[i] for i in range(len(vars))}
				try:
					f_val = self.eval(env)
					if isinstance(f_val, Rational):
						f_val = f_val.numerator() / f_val.denominator()
					f_val = float(f_val)
				except Exception:
					return None
				# Check convergence
				if abs(f_val) < tol:
					return {vars[i]: x[i] for i in range(len(vars))}
				# Compute Jacobian (gradient for scalar function)
				jacobian = []
				for var in vars:
					deriv = self._get_derivative(var)
					if deriv is None:
						return None
					try:
						df_dxi = deriv.eval(env)
						if isinstance(df_dxi, Rational):
							df_dxi = df_dxi.numerator() / df_dxi.denominator()
						jacobian.append(float(df_dxi))
					except Exception:
						return None
				# For scalar function f: Rⁿ → R, we have gradient, not full Jacobian
				# Use gradient descent-style update: x_new = x - (f / ||grad||²) * grad
				grad_norm_sq = sum(g * g for g in jacobian)
				if grad_norm_sq < 1e-15:
					# Gradient is zero, can't proceed
					return None
				# Update: x_new = x - (f / ||grad||²) * grad
				step = f_val / grad_norm_sq
				for i in range(len(vars)):
					x[i] = x[i] - step * jacobian[i]
			# Check final value
			env = {vars[i]: x[i] for i in range(len(vars))}
			try:
				final_val = self.eval(env)
				if isinstance(final_val, Rational):
					final_val = final_val.numerator() / final_val.denominator()
				if abs(final_val) < tol:
					return {vars[i]: x[i] for i in range(len(vars))}
			except Exception:
				pass
			return None
		def find_multivariate_roots_system(self, equations: list['CAS.ExprResult'], vars: list[str], initial_guess: Dict[str, float], tol: float = 1e-8, max_iter: int = 100) -> Optional[Dict[str, float]]:
			"""Solve a system of equations using multivariate Newton's method.
			
			Args:
				equations: List of ExprResult objects representing f₁=0, f₂=0, ..., fₙ=0
				vars: List of variable names (must match number of equations)
				initial_guess: Dictionary mapping variable names to initial values
				tol: Tolerance for convergence
				max_iter: Maximum iterations
			
			Returns:
				Dictionary mapping variables to solution values, or None if not converged
			"""
			if len(equations) != len(vars):
				raise ValueError("Number of equations must match number of variables")
			if len(equations) == 0:
				return {}
			# Multivariate Newton's method for systems
			x = [initial_guess.get(v, 0.0) for v in vars]
			for iteration in range(max_iter):
				# Evaluate all functions at current point
				env = {vars[i]: x[i] for i in range(len(vars))}
				f_vec = []
				for eq in equations:
					try:
						f_val = eq.eval(env)
						if isinstance(f_val, Rational):
							f_val = f_val.numerator() / f_val.denominator()
						f_vec.append(float(f_val))
					except Exception:
						return None
				# Check convergence
				if all(abs(f) < tol for f in f_vec):
					return {vars[i]: x[i] for i in range(len(vars))}
				# Compute Jacobian matrix
				jacobian = []
				for eq in equations:
					row = []
					for var in vars:
						deriv = eq._get_derivative(var)
						if deriv is None:
							return None
						try:
							df_dxi = deriv.eval(env)
							if isinstance(df_dxi, Rational):
								df_dxi = df_dxi.numerator() / df_dxi.denominator()
							row.append(float(df_dxi))
						except Exception:
							return None
					jacobian.append(row)
				# Solve J * delta = -f for delta using Gaussian elimination (simple case)
				# For now, use a simple iterative update: delta = -J^(-1) * f
				# For 2x2 or 3x3, we can invert directly
				n = len(vars)
				if n == 1:
					# 1D case
					if abs(jacobian[0][0]) < 1e-15:
						return None
					delta = [-f_vec[0] / jacobian[0][0]]
				elif n == 2:
					# 2x2 case: solve directly
					det = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0]
					if abs(det) < 1e-15:
						return None
					delta = [
						(-f_vec[0] * jacobian[1][1] + f_vec[1] * jacobian[0][1]) / det,
						(jacobian[0][0] * f_vec[1] - jacobian[1][0] * f_vec[0]) / det
					]
				else:
					# For higher dimensions, solve J * delta = -f using Gaussian elimination
					# Build augmented matrix [J | -f]
					aug = [[0.0] * (n + 1) for _ in range(n)]
					for i in range(n):
						for j in range(n):
							aug[i][j] = jacobian[i][j]
						aug[i][n] = -f_vec[i]
					# Forward elimination
					for i in range(n):
						# Find pivot
						max_row = i
						for k in range(i + 1, n):
							if abs(aug[k][i]) > abs(aug[max_row][i]):
								max_row = k
						aug[i], aug[max_row] = aug[max_row], aug[i]
						if abs(aug[i][i]) < 1e-15:
							return None  # Singular matrix
						# Eliminate
						for k in range(i + 1, n):
							factor = aug[k][i] / aug[i][i]
							for j in range(i, n + 1):
								aug[k][j] -= factor * aug[i][j]
					# Back substitution
					delta = [0.0] * n
					for i in range(n - 1, -1, -1):
						delta[i] = aug[i][n]
						for j in range(i + 1, n):
							delta[i] -= aug[i][j] * delta[j]
						delta[i] /= aug[i][i]
				# Update solution
				for i in range(len(vars)):
					x[i] = x[i] + delta[i]
			# Check final values
			env = {vars[i]: x[i] for i in range(len(vars))}
			f_vec = []
			for eq in equations:
				try:
					f_val = eq.eval(env)
					if isinstance(f_val, Rational):
						f_val = f_val.numerator() / f_val.denominator()
					f_vec.append(float(f_val))
				except Exception:
					return None
			if all(abs(f) < tol for f in f_vec):
				return {vars[i]: x[i] for i in range(len(vars))}
			return None
	def eval(self, expr: Any, env: Dict[str, NumberLike] | None = None) -> Any:
		# If expression contains a function-call pattern like 'name(...)', use EDAG-based evaluator
		if isinstance(expr, CAS.ExprResult):
			return expr.eval(env or {})
		if isinstance(expr, EDAG):
			return expr.eval(env or {})
		if isinstance(expr, Polynomial):
			env_r = {k: (v if isinstance(v, Rational) else Rational(v,1) if isinstance(v,int) else Rational(int(v*10**6),10**6)) for k,v in (env or {}).items()}
			return expr.eval(env_r)
		if re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\(", expr):
			dag = parse_expression_edag(expr)
			return dag.eval(env or {})
		# Try polynomial path
		try:
			p = parse_polynomial(expr)
			env_r = {k: (v if isinstance(v, Rational) else Rational(v,1) if isinstance(v,int) else Rational(int(v*10**6),10**6)) for k,v in (env or {}).items()}
			return p.eval(env_r)
		except Exception:
			# Fallback to EDAG-based evaluator for general expressions
			dag = parse_expression_edag(expr)
			return dag.eval(env or {})
	def differentiate(self, expr: Any, var: str) -> 'CAS.ExprResult':
		# Already parsed inputs
		if isinstance(expr, CAS.ExprResult):
			if expr._poly is not None:
				return CAS.ExprResult(poly=expr._poly.derivative(var))
			if expr._dag is not None:
				return CAS.ExprResult(dag=expr._dag.derivative(var))
		if isinstance(expr, Polynomial):
			return CAS.ExprResult(poly=expr.derivative(var))
		if isinstance(expr, EDAG):
			return CAS.ExprResult(dag=expr.derivative(var))
		# Prefer EDAG if we detect function calls or decimals
		if isinstance(expr, str) and (re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\(", expr) or re.search(r"\d+\.\d+", expr)):
			dag = parse_expression_edag(expr)
			ddag = dag.derivative(var)
			return CAS.ExprResult(dag=ddag)
		# Try polynomial path; if it fails (e.g., e^x, x^(1/2)), fall back to EDAG
		try:
			p = parse_polynomial(expr if isinstance(expr,str) else str(expr))
			return CAS.ExprResult(poly=p.derivative(var))
		except Exception:
			dag = parse_expression_edag(expr if isinstance(expr,str) else str(expr))
			ddag = dag.derivative(var)
			return CAS.ExprResult(dag=ddag)
	def integrate(self, expr: Any, var: str) -> 'CAS.ExprResult':
		# Already parsed inputs
		if isinstance(expr, CAS.ExprResult):
			if expr._poly is not None:
				return CAS.ExprResult(poly=expr._poly.integral(var))
			if expr._dag is not None:
				return CAS.ExprResult(dag=expr._dag.integral(var))
		if isinstance(expr, Polynomial):
			return CAS.ExprResult(poly=expr.integral(var))
		if isinstance(expr, EDAG):
			return CAS.ExprResult(dag=expr.integral(var))
		# Prefer EDAG if we detect function calls or decimals
		if isinstance(expr, str) and (re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\(", expr) or re.search(r"\d+\.\d+", expr)):
			dag = parse_expression_edag(expr)
			I = dag.integral(var)
			return CAS.ExprResult(dag=I)
		# Otherwise try polynomial first, then EDAG
		try:
			p = parse_polynomial(expr if isinstance(expr,str) else str(expr))
			return CAS.ExprResult(poly=p.integral(var))
		except Exception:
			dag = parse_expression_edag(expr if isinstance(expr,str) else str(expr))
			I = dag.integral(var)
			return CAS.ExprResult(dag=I)
	def simplify(self, expr: str) -> str:
		p = parse_polynomial(expr)
		p.normalize()
		return str(p)
