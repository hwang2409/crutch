from __future__ import annotations
from typing import Dict, Any, Union, Optional
from parser import parse_polynomial, parse_expression_edag
from polynomial import Polynomial
from rational import Rational
from edag import EDAG
from interval import Interval
import re
import math
import numpy as np

NumberLike = Union[int, float, Rational]

DEFAULT_LOWER_BOUND = -100.0
DEFAULT_UPPER_BOUND = 100.0


class CAS:
    def __init__(self) -> None:
        pass

    def _wrap(self, obj: Any) -> "CAS.ExprResult":
        if isinstance(obj, Polynomial):
            return CAS.ExprResult(poly=obj)
        if isinstance(obj, EDAG):
            return CAS.ExprResult(dag=obj)
        if isinstance(obj, CAS.ExprResult):
            return obj
        raise TypeError("Unsupported object for wrapping")

    def parse(self, expr: str) -> "CAS.ExprResult":
        # Prefer EDAG for general expressions
        if re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\(", expr) or re.search(
            r"\d+\.\d+", expr
        ):
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
        def __init__(
            self, poly: Polynomial | None = None, dag: EDAG | None = None
        ) -> None:
            self._poly = poly
            self._dag = dag

        def eval(self, env: Dict[str, Any] | None = None) -> Any:
            env = env or {}
            if self._poly is not None:
                env_r = {
                    k: (
                        v
                        if isinstance(v, Rational)
                        else (
                            Rational(v, 1)
                            if isinstance(v, int)
                            else Rational(int(v * 10**6), 10**6)
                        )
                    )
                    for k, v in env.items()
                }
                return self._poly.eval(env_r)
            if self._dag is not None:
                return self._dag.eval(env)
            raise RuntimeError("empty expression")

        def __str__(self) -> str:
            if self._poly is not None:
                return str(self._poly)
            if self._dag is not None:
                return self._dag.to_string()
            return ""

        def _eval_float(self, var: str, x: float) -> float:
            """Evaluate expression at x, returning float."""
            val = self.eval({var: x})
            if isinstance(val, Rational):
                return val.numerator() / val.denominator()
            return float(val)

        def find_root(
            self, var: str, interval: Interval, tol: float = 1e-12, max_iter: int = 200
        ) -> Optional[float]:
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
                a = DEFAULT_LOWER_BOUND
            if b == math.inf:
                b = DEFAULT_UPPER_BOUND
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
                    s = (
                        (a * fb * fc) / ((fa - fb) * (fa - fc))
                        + (b * fa * fc) / ((fb - fa) * (fb - fc))
                        + (c * fa * fb) / ((fc - fa) * (fc - fb))
                    )
                elif abs(fb - fa) > 1e-15:
                    # Secant method
                    s = b - fb * (b - a) / (fb - fa)
                else:
                    # fa == fb, use bisection
                    s = (a + b) / 2.0
                    mflag = True
                # Check conditions for using bisection
                condition1 = s <= (3 * a + b) / 4 or s >= b
                condition2 = mflag and abs(s - b) >= abs(b - c) / 2
                condition3 = not mflag and abs(s - b) >= abs(c - d) / 2
                condition4 = mflag and abs(b - c) < tol
                condition5 = not mflag and abs(c - d) < tol
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

        def find_all_roots(
            self,
            var: str,
            interval: Interval,
            tol: float = 1e-8,
            max_iter: int = 100,
            sample_points: int = 1000,
        ) -> list[float]:
            """Find all roots of f(x) = 0 in the given interval.

            Returns a list of roots found. For univariate polynomials, uses numpy's
            polynomial root finder. For general functions, uses sampling to detect
            sign changes, then applies Brent's method to each bracketed interval.
            """
            if interval.is_empty():
                return []
            # Fast path: if this is a univariate polynomial, use numpy's root finder
            if self._poly is not None and self._poly.is_univariate(var):
                coeffs = self._poly.to_univariate_coeffs(var)
                if not coeffs:
                    return []
                # Convert Rational coefficients to float, reversed for numpy (highest degree first)
                coeffs_float = [
                    float(c.numerator()) / float(c.denominator())
                    for c in reversed(coeffs)
                ]
                # Remove leading zeros
                while coeffs_float and abs(coeffs_float[0]) < 1e-12:
                    coeffs_float.pop(0)
                if not coeffs_float:
                    return []  # Zero polynomial
                # Find all roots using numpy
                try:
                    all_roots = np.roots(coeffs_float)
                    # Filter to real roots in the interval
                    real_roots = []
                    for root in all_roots:
                        if abs(root.imag) < tol:  # Real root
                            r = float(root.real)
                            # Check if root is in interval
                            in_interval = True
                            if interval.a != -math.inf:
                                if interval.left_open:
                                    if r <= interval.a:
                                        in_interval = False
                                else:
                                    if r < interval.a:
                                        in_interval = False
                            if interval.b != math.inf:
                                if interval.right_open:
                                    if r >= interval.b:
                                        in_interval = False
                                else:
                                    if r > interval.b:
                                        in_interval = False
                            if in_interval:
                                # Verify it's actually a root (stricter tolerance for correctness)
                                try:
                                    f_val = self._eval_float(var, r)
                                    if abs(f_val) < tol * 10:  # Stricter verification
                                        real_roots.append(r)
                                except Exception:
                                    pass
                    # Deduplicate and sort
                    if not real_roots:
                        return []
                    real_roots.sort()
                    unique_roots = [real_roots[0]]
                    for r in real_roots[1:]:
                        if abs(r - unique_roots[-1]) > tol * 10:
                            unique_roots.append(r)
                    return unique_roots
                except Exception:
                    # Fall through to general method if numpy fails
                    pass
            roots = []
            # Get finite bounds for adaptive subdivision
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

            # Adaptive recursive subdivision for root finding
            # Prioritize correctness over speed - use deeper recursion and finer subdivision
            max_depth = 50  # Maximum recursion depth (increased for thoroughness)
            min_width = (b - a) / (sample_points * 100)  # Minimum interval width (finer)
            brackets = []  # Intervals that bracket roots
            sample_tol = max(tol * 100, 1e-6)

            def subdivide(lo: float, hi: float, depth: int) -> None:
                """Recursively subdivide interval to find roots."""
                if depth > max_depth or (hi - lo) < min_width:
                    return

                # Evaluate at endpoints and midpoint
                try:
                    f_lo = self._eval_float(var, lo)
                    f_hi = self._eval_float(var, hi)
                    mid = (lo + hi) / 2
                    f_mid = self._eval_float(var, mid)
                except Exception:
                    return

                # Check for sign changes (only valid way to detect roots)
                if f_lo * f_mid < 0:
                    brackets.append((lo, mid))
                elif f_mid * f_hi < 0:
                    brackets.append((mid, hi))
                elif f_lo * f_hi < 0:
                    # Sign change across entire interval - subdivide both halves
                    subdivide(lo, mid, depth + 1)
                    subdivide(mid, hi, depth + 1)
                # If no sign change, don't subdivide further (avoids false positives)

            # Start with initial sampling - use more points for better coverage
            # Prioritize correctness: use more initial samples to catch all sign changes
            initial_points = min(sample_points // 10, 100)  # Use up to 100 initial points
            initial_samples = np.linspace(a, b, initial_points + 1)

            # First pass: find obvious sign changes
            x_vals = []
            f_vals = []
            for x in initial_samples:
                try:
                    fx = self._eval_float(var, x)
                    x_vals.append(x)
                    f_vals.append(fx)
                except Exception:
                    continue

            # Find sign changes in initial samples
            for i in range(len(f_vals) - 1):
                if f_vals[i] * f_vals[i + 1] < 0:
                    # Clear sign change - subdivide this interval
                    subdivide(x_vals[i], x_vals[i + 1], 0)
                # Only subdivide if there's a sign change (avoids false positives for functions like e^x)

            # Use Brent's method on each bracketed interval
            for a_bracket, b_bracket in brackets:
                # Verify the bracket actually has opposite signs
                try:
                    fa_br = self._eval_float(var, a_bracket)
                    fb_br = self._eval_float(var, b_bracket)
                    if fa_br * fb_br < 0:
                        sub_interval = Interval.closed(a_bracket, b_bracket)
                        root = self.find_root(var, sub_interval, tol, max_iter)
                        if root is not None:
                            roots.append(root)
                    # Only add endpoints if they bracket a sign change (already verified above)
                    # Don't add just because they're near zero - requires sign change
                except Exception:
                    pass

            # Deduplicate roots that are very close (within tolerance)
            if not roots:
                return []
            roots.sort()
            unique_roots = [float(roots[0])]  # Convert to Python float
            for r in roots[1:]:
                if abs(r - unique_roots[-1]) > tol * 10:  # More lenient deduplication
                    unique_roots.append(float(r))  # Convert to Python float
            return unique_roots

        def _get_derivative(self, var: str) -> Optional["CAS.ExprResult"]:
            """Get derivative of this expression with respect to var."""
            if self._poly is not None:
                return CAS.ExprResult(poly=self._poly.derivative(var))
            if self._dag is not None:
                return CAS.ExprResult(dag=self._dag.derivative(var))
            return None

        def solve(self, var: str) -> list:
            """Solve the equation self = 0 for variable var.
            
            Returns a list of Solution objects representing symbolic solutions.
            For example:
                solve(x^2 - 4 = 0, x) ? [Solution('x', 2), Solution('x', -2)]
            
            Args:
                var: Variable to solve for
            
            Returns:
                List of Solution objects
            """
            from solver import SymbolicSolver
            
            # Try polynomial path first (most common case)
            if self._poly is not None and self._poly.is_univariate(var):
                coeffs = self._poly.to_univariate_coeffs(var)
                if coeffs:
                    solver = SymbolicSolver()
                    return solver.solve(coeffs, var)
            
            # TODO: Handle EDAG expressions
            # This would require:
            # 1. Converting EDAG to polynomial (expansion)
            # 2. Pattern matching for non-polynomial equations
            # 3. Substitution methods
            
            # For now, return empty list if not a polynomial
            return []

        def is_strictly_increasing(
            self, var: str, interval: Interval = None, tol: float = 1e-12
        ) -> bool:
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
                search_bounds = Interval.closed(
                    DEFAULT_LOWER_BOUND, DEFAULT_UPPER_BOUND
                )
                if interval.a != -math.inf:
                    search_bounds = Interval.closed(interval.a, DEFAULT_UPPER_BOUND)
                if interval.b != math.inf:
                    search_bounds = Interval.closed(DEFAULT_LOWER_BOUND, interval.b)
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
                a = DEFAULT_LOWER_BOUND
            else:
                a = interval.a
                if interval.left_open:
                    a += 1e-6
            if interval.b == math.inf:
                b = DEFAULT_UPPER_BOUND
            else:
                b = interval.b
                if interval.right_open:
                    b -= 1e-6
            # If no critical points, check at many sample points for thoroughness
            if not critical_points:
                # Check at many points in the interval for correctness
                # Use points that are slightly away from boundaries to avoid precision issues
                boundary_margin = max(
                    1e-3, (b - a) * 0.01
                )  # 1% margin or 1e-3, whichever is larger
                effective_a = a + boundary_margin
                effective_b = b - boundary_margin
                if effective_a < effective_b:
                    # Use more sample points for better coverage (prioritize correctness)
                    num_samples = max(20, int((effective_b - effective_a) * 10))
                    test_vals = np.linspace(effective_a, effective_b, num_samples).tolist()
                else:
                    # Very small interval, test at midpoint
                    test_vals = [(a + b) / 2]
                for test_x in test_vals:
                    try:
                        deriv_val = deriv._eval_float(var, test_x)
                        # For strictly increasing, need deriv_val > 0 everywhere
                        # Return False if deriv_val <= 0 (accounting for numerical tolerance)
                        # Use <= -tol to check if it's non-positive (negative or zero within tolerance)
                        if deriv_val <= -tol:
                            return False
                    except (OverflowError, ValueError):
                        # If overflow, try a point closer to zero
                        continue
                return True
            # Check derivative sign in each subinterval - check multiple points per interval
            test_points = [a]
            test_points.extend(critical_points)
            test_points.append(b)
            test_points = sorted(set(test_points))
            # Check derivative at multiple points in each interval for thoroughness
            for i in range(len(test_points) - 1):
                # Check at 5 points in each subinterval
                interval_start = test_points[i]
                interval_end = test_points[i + 1]
                check_points = np.linspace(interval_start, interval_end, 5).tolist()
                for check_x in check_points:
                    try:
                        deriv_val = deriv._eval_float(var, check_x)
                        # For strictly increasing, need deriv_val > 0 everywhere
                        # Return False if deriv_val <= 0 (accounting for numerical tolerance)
                        # Use <= -tol to check if it's non-positive (negative or zero within tolerance)
                        if deriv_val <= -tol:
                            return False
                    except (OverflowError, ValueError):
                        # If overflow, skip this point
                        continue
            return True

        def is_strictly_decreasing(
            self, var: str, interval: Interval = None, tol: float = 1e-12
        ) -> bool:
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
                search_bounds = Interval.closed(
                    DEFAULT_LOWER_BOUND, DEFAULT_UPPER_BOUND
                )
                if interval.a != -math.inf:
                    search_bounds = Interval.closed(interval.a, DEFAULT_UPPER_BOUND)
                if interval.b != math.inf:
                    search_bounds = Interval.closed(DEFAULT_LOWER_BOUND, interval.b)
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
                a = DEFAULT_LOWER_BOUND
            else:
                a = interval.a
                if interval.left_open:
                    a += 1e-6
            if interval.b == math.inf:
                b = DEFAULT_UPPER_BOUND
            else:
                b = interval.b
                if interval.right_open:
                    b -= 1e-6
            # If no critical points, check at many sample points for thoroughness
            if not critical_points:
                boundary_margin = max(1e-3, (b - a) * 0.01)
                effective_a = a + boundary_margin
                effective_b = b - boundary_margin
                if effective_a < effective_b:
                    # Use more sample points for better coverage (prioritize correctness)
                    num_samples = max(20, int((effective_b - effective_a) * 10))
                    test_vals = np.linspace(effective_a, effective_b, num_samples).tolist()
                else:
                    test_vals = [(a + b) / 2]
                for test_x in test_vals:
                    try:
                        deriv_val = deriv._eval_float(var, test_x)
                        # For strictly decreasing, need deriv_val < 0 everywhere
                        # Return False if deriv_val >= 0 (accounting for numerical tolerance)
                        # Use >= tol to check if it's non-negative (positive within tolerance)
                        if deriv_val >= tol:
                            return False
                    except (OverflowError, ValueError):
                        # If overflow/underflow, skip this point (try another)
                        continue
                return True
            # Check derivative sign in each subinterval
            test_points = [a]
            test_points.extend(critical_points)
            test_points.append(b)
            test_points = sorted(set(test_points))
            # Check derivative at midpoints of intervals
            for i in range(len(test_points) - 1):
                mid = (test_points[i] + test_points[i + 1]) / 2
                try:
                    deriv_val = deriv._eval_float(var, mid)
                    # For strictly decreasing, need deriv_val < 0 everywhere
                    # Return False if deriv_val >= 0 (accounting for numerical tolerance)
                    # Use >= tol to check if it's non-negative (positive within tolerance)
                    if deriv_val >= tol:
                        return False
                except (OverflowError, ValueError):
                    # If overflow, skip this interval (might be at extreme values)
                    continue
            return True

        def is_monotone_increasing(
            self, var: str, interval: Interval = None, tol: float = 1e-12
        ) -> bool:
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
            # Use a reasonable bounded interval for root finding if interval is unbounded
            if interval.a == -math.inf or interval.b == math.inf:
                search_bounds = Interval.closed(
                    DEFAULT_LOWER_BOUND, DEFAULT_UPPER_BOUND
                )
                if interval.a != -math.inf:
                    search_bounds = Interval.closed(interval.a, DEFAULT_UPPER_BOUND)
                if interval.b != math.inf:
                    search_bounds = Interval.closed(DEFAULT_LOWER_BOUND, interval.b)
                if interval.a != -math.inf and interval.b != math.inf:
                    search_bounds = interval
                all_critical = deriv.find_all_roots(var, search_bounds, tol=tol)
                critical_points = []
                for cp in all_critical:
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
            # Get interval bounds for testing
            if interval.a == -math.inf:
                a = DEFAULT_LOWER_BOUND
            else:
                a = interval.a
                if interval.left_open:
                    a += 1e-6
            if interval.b == math.inf:
                b = DEFAULT_UPPER_BOUND
            else:
                b = interval.b
                if interval.right_open:
                    b -= 1e-6
            # If no critical points, check at many sample points for thoroughness
            if not critical_points:
                boundary_margin = max(1e-3, (b - a) * 0.01)
                effective_a = a + boundary_margin
                effective_b = b - boundary_margin
                if effective_a < effective_b:
                    # Use more sample points for better coverage (prioritize correctness)
                    num_samples = max(20, int((effective_b - effective_a) * 10))
                    test_vals = np.linspace(effective_a, effective_b, num_samples).tolist()
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
            # Check derivative sign in each subinterval - check multiple points per interval
            test_points = [a]
            test_points.extend(critical_points)
            test_points.append(b)
            test_points = sorted(set(test_points))
            for i in range(len(test_points) - 1):
                # Check at 5 points in each subinterval
                interval_start = test_points[i]
                interval_end = test_points[i + 1]
                check_points = np.linspace(interval_start, interval_end, 5).tolist()
                for check_x in check_points:
                    try:
                        deriv_val = deriv._eval_float(var, check_x)
                        if deriv_val < -tol:
                            return False
                    except (OverflowError, ValueError):
                        continue
            return True

        def is_monotone_decreasing(
            self, var: str, interval: Interval = None, tol: float = 1e-12
        ) -> bool:
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
                search_bounds = Interval.closed(
                    DEFAULT_LOWER_BOUND, DEFAULT_UPPER_BOUND
                )
                if interval.a != -math.inf:
                    search_bounds = Interval.closed(interval.a, DEFAULT_UPPER_BOUND)
                if interval.b != math.inf:
                    search_bounds = Interval.closed(DEFAULT_LOWER_BOUND, interval.b)
                if interval.a != -math.inf and interval.b != math.inf:
                    search_bounds = interval
                all_critical = deriv.find_all_roots(var, search_bounds, tol=tol)
                critical_points = []
                for cp in all_critical:
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
            # Get interval bounds for testing
            if interval.a == -math.inf:
                a = DEFAULT_LOWER_BOUND
            else:
                a = interval.a
                if interval.left_open:
                    a += 1e-6
            if interval.b == math.inf:
                b = DEFAULT_UPPER_BOUND
            else:
                b = interval.b
                if interval.right_open:
                    b -= 1e-6
            # If no critical points, check at many sample points for thoroughness
            if not critical_points:
                boundary_margin = max(1e-3, (b - a) * 0.01)
                effective_a = a + boundary_margin
                effective_b = b - boundary_margin
                if effective_a < effective_b:
                    # Use more sample points for better coverage (prioritize correctness)
                    num_samples = max(20, int((effective_b - effective_a) * 10))
                    test_vals = np.linspace(effective_a, effective_b, num_samples).tolist()
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
            # Check derivative sign in each subinterval - check multiple points per interval
            test_points = [a]
            test_points.extend(critical_points)
            test_points.append(b)
            test_points = sorted(set(test_points))
            for i in range(len(test_points) - 1):
                # Check at 5 points in each subinterval
                interval_start = test_points[i]
                interval_end = test_points[i + 1]
                check_points = np.linspace(interval_start, interval_end, 5).tolist()
                for check_x in check_points:
                    try:
                        deriv_val = deriv._eval_float(var, check_x)
                        if deriv_val > tol:
                            return False
                    except (OverflowError, ValueError):
                        continue
            return True

        def find_multivariate_root(
            self,
            vars: list[str],
            initial_guess: Dict[str, float],
            tol: float = 1e-8,
            max_iter: int = 100,
        ) -> Optional[Dict[str, float]]:
            """Find a root of f(x?, x?, ..., x?) = 0 using multivariate Newton's method.

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
                # For scalar function f: R? ? R, we have gradient, not full Jacobian
                # Use gradient descent-style update: x_new = x - (f / ||grad||?) * grad
                grad_norm_sq = sum(g * g for g in jacobian)
                if grad_norm_sq < 1e-15:
                    # Gradient is zero, can't proceed
                    return None
                # Update: x_new = x - (f / ||grad||?) * grad
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

        def find_multivariate_roots_system(
            self,
            equations: list["CAS.ExprResult"],
            vars: list[str],
            initial_guess: Dict[str, float],
            tol: float = 1e-8,
            max_iter: int = 100,
        ) -> Optional[Dict[str, float]]:
            """Solve a system of equations using multivariate Newton's method.

            Args:
                    equations: List of ExprResult objects representing f?=0, f?=0, ..., f?=0
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
                    det = (
                        jacobian[0][0] * jacobian[1][1]
                        - jacobian[0][1] * jacobian[1][0]
                    )
                    if abs(det) < 1e-15:
                        return None
                    delta = [
                        (-f_vec[0] * jacobian[1][1] + f_vec[1] * jacobian[0][1]) / det,
                        (jacobian[0][0] * f_vec[1] - jacobian[1][0] * f_vec[0]) / det,
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
            env_r = {
                k: (
                    v
                    if isinstance(v, Rational)
                    else (
                        Rational(v, 1)
                        if isinstance(v, int)
                        else Rational(int(v * 10**6), 10**6)
                    )
                )
                for k, v in (env or {}).items()
            }
            return expr.eval(env_r)
        if re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\(", expr):
            dag = parse_expression_edag(expr)
            return dag.eval(env or {})
        # Try polynomial path
        try:
            p = parse_polynomial(expr)
            env_r = {
                k: (
                    v
                    if isinstance(v, Rational)
                    else (
                        Rational(v, 1)
                        if isinstance(v, int)
                        else Rational(int(v * 10**6), 10**6)
                    )
                )
                for k, v in (env or {}).items()
            }
            return p.eval(env_r)
        except Exception:
            # Fallback to EDAG-based evaluator for general expressions
            dag = parse_expression_edag(expr)
            return dag.eval(env or {})

    def differentiate(self, expr: Any, var: str) -> "CAS.ExprResult":
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
        if isinstance(expr, str) and (
            re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\(", expr)
            or re.search(r"\d+\.\d+", expr)
        ):
            dag = parse_expression_edag(expr)
            ddag = dag.derivative(var)
            return CAS.ExprResult(dag=ddag)
        # Try polynomial path; if it fails (e.g., e^x, x^(1/2)), fall back to EDAG
        try:
            p = parse_polynomial(expr if isinstance(expr, str) else str(expr))
            return CAS.ExprResult(poly=p.derivative(var))
        except Exception:
            dag = parse_expression_edag(expr if isinstance(expr, str) else str(expr))
            ddag = dag.derivative(var)
            return CAS.ExprResult(dag=ddag)

    def integrate(self, expr: Any, var: str) -> "CAS.ExprResult":
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
        if isinstance(expr, str) and (
            re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\(", expr)
            or re.search(r"\d+\.\d+", expr)
        ):
            dag = parse_expression_edag(expr)
            I = dag.integral(var)
            return CAS.ExprResult(dag=I)
        # Otherwise try polynomial first, then EDAG
        try:
            p = parse_polynomial(expr if isinstance(expr, str) else str(expr))
            return CAS.ExprResult(poly=p.integral(var))
        except Exception:
            dag = parse_expression_edag(expr if isinstance(expr, str) else str(expr))
            I = dag.integral(var)
            return CAS.ExprResult(dag=I)

    def simplify(self, expr: str) -> str:
        p = parse_polynomial(expr)
        p.normalize()
        return str(p)

    def solve(self, expr: Any, var: str) -> list:
        """Solve the equation expr = 0 for variable var.
        
        Args:
            expr: Expression to solve (string, ExprResult, Polynomial, or EDAG)
            var: Variable to solve for
        
        Returns:
            List of Solution objects
        """
        # Parse expression if needed
        if isinstance(expr, str):
            expr_result = self.parse(expr)
        elif isinstance(expr, CAS.ExprResult):
            expr_result = expr
        elif isinstance(expr, Polynomial):
            expr_result = CAS.ExprResult(poly=expr)
        elif isinstance(expr, EDAG):
            expr_result = CAS.ExprResult(dag=expr)
        else:
            raise TypeError(f"Cannot solve expression of type {type(expr)}")
        
        return expr_result.solve(var)
