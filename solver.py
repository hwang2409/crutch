"""
Symbolic Solver Module

This module provides symbolic equation solving capabilities.
Starts with linear and quadratic equations, expanding to more complex cases.
"""
from __future__ import annotations
from typing import List, Union, Optional
from dataclasses import dataclass
from rational import Rational
from polynomial import Polynomial
from monomial import Monomial


@dataclass
class Solution:
    """Represents a solution to an equation."""
    variable: str
    value: Union[Rational, "SymbolicExpression", None]
    multiplicity: int = 1

    def __str__(self) -> str:
        if self.value is None:
            if self.multiplicity == float('inf'):
                return f"{self.variable} ? ?"
            elif self.multiplicity == 0:
                return "No solution"
            else:
                return f"{self.variable} = ?"
        
        # Format the solution value
        value_str = str(self.value)
        
        # Add multiplicity if > 1
        if self.multiplicity > 1:
            return f"{self.variable} = {value_str} (multiplicity {self.multiplicity})"
        else:
            return f"{self.variable} = {value_str}"
    
    @staticmethod
    def all_reals(var: str) -> "Solution":
        """Represents x ? ? (infinite solutions)."""
        return Solution(var, None, multiplicity=float('inf'))
    
    @staticmethod
    def no_solution(var: str) -> "Solution":
        """Represents no solution."""
        return Solution(var, None, multiplicity=0)


class SymbolicExpression:
    """Base class for symbolic expressions that can't be simplified to a number."""
    pass


class QuadraticRoot(SymbolicExpression):
    """Represents (-b ? ?(b?-4ac))/(2a) symbolically."""
    def __init__(self, a: Rational, b: Rational, c: Rational, positive: bool):
        self.a = a
        self.b = b
        self.c = c
        self.positive = positive
    
    def __str__(self) -> str:
        sign = "+" if self.positive else "-"
        discriminant = self.b * self.b - Rational(4, 1) * self.a * self.c
        
        # Simplify the radical in the discriminant
        sqrt_coeff, sqrt_radicand = self._simplify_radical(discriminant)
        
        # Format numerator
        if self.b.is_zero():
            if sqrt_coeff == Rational(1, 1):
                numerator = f"0 {sign} sqrt({sqrt_radicand})"
            elif sqrt_coeff.denominator() == 1:
                if sqrt_coeff.numerator() == 1:
                    numerator = f"0 {sign} sqrt({sqrt_radicand})"
                else:
                    numerator = f"0 {sign} {sqrt_coeff.numerator()}*sqrt({sqrt_radicand})"
            else:
                numerator = f"0 {sign} {sqrt_coeff}*sqrt({sqrt_radicand})"
        else:
            # Compute -b as a Rational to avoid double negatives in string formatting
            neg_b = Rational(-self.b.numerator(), self.b.denominator())
            if sqrt_coeff == Rational(1, 1):
                numerator = f"({neg_b} {sign} sqrt({sqrt_radicand}))"
            elif sqrt_coeff.denominator() == 1:
                if sqrt_coeff.numerator() == 1:
                    numerator = f"({neg_b} {sign} sqrt({sqrt_radicand}))"
                else:
                    numerator = f"({neg_b} {sign} {sqrt_coeff.numerator()}*sqrt({sqrt_radicand}))"
            else:
                numerator = f"({neg_b} {sign} {sqrt_coeff}*sqrt({sqrt_radicand}))"
        
        # Format denominator: calculate 2*a and simplify
        two_a = Rational(2, 1) * self.a
        if two_a == Rational(1, 1):
            denominator = "1"
        elif two_a.denominator() == 1:
            denominator = str(two_a.numerator())
        else:
            denominator = str(two_a)
        
        # Simplify the fraction: if sqrt_coeff and denominator have common factors, simplify
        if sqrt_coeff.denominator() == 1 and two_a.denominator() == 1:
            # Both are integers, try to simplify
            gcd_coeff = self._gcd(abs(sqrt_coeff.numerator()), abs(two_a.numerator()))
            if gcd_coeff > 1:
                simplified_coeff = Rational(sqrt_coeff.numerator() // gcd_coeff, 1)
                simplified_denom = Rational(two_a.numerator() // gcd_coeff, 1)
                if simplified_coeff == Rational(1, 1):
                    # Can cancel out the coefficient
                    if self.b.is_zero():
                        numerator = f"0 {sign} sqrt({sqrt_radicand})"
                    else:
                        numerator = f"({neg_b} {sign} sqrt({sqrt_radicand}))"
                else:
                    if self.b.is_zero():
                        numerator = f"0 {sign} {simplified_coeff.numerator()}*sqrt({sqrt_radicand})"
                    else:
                        numerator = f"({neg_b} {sign} {simplified_coeff.numerator()}*sqrt({sqrt_radicand}))"
                
                # Format denominator - skip if it's 1
                if simplified_denom == Rational(1, 1):
                    return numerator
                else:
                    return f"{numerator}/{simplified_denom.numerator()}"
        
        # No simplification possible, format normally
        if two_a == Rational(1, 1):
            return numerator  # No denominator needed
        elif two_a.denominator() == 1:
            return f"{numerator}/{two_a.numerator()}"
        else:
            return f"{numerator}/{two_a}"
    
    def _simplify_radical(self, r: Rational) -> tuple[Rational, Rational]:
        """Helper to simplify radicals, delegating to solver instance if available."""
        # This is a bit awkward - we need access to the solver's simplify_radical method
        # For now, implement a simplified version here
        if r < Rational(0, 1):
            return (Rational(1, 1), r)
        
        if r.is_zero():
            return (Rational(0, 1), Rational(0, 1))
        
        num = abs(r.numerator())
        den = abs(r.denominator())
        
        # Factor perfect squares from numerator
        num_coeff = 1
        num_radicand = num
        
        for square in [4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225]:
            if num_radicand % square == 0:
                sqrt_val = int(square ** 0.5)
                num_coeff *= sqrt_val
                num_radicand //= square
                while num_radicand % square == 0:
                    num_coeff *= sqrt_val
                    num_radicand //= square
        
        # Factor perfect squares from denominator
        den_coeff = 1
        den_radicand = den
        
        for square in [4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225]:
            if den_radicand % square == 0:
                sqrt_val = int(square ** 0.5)
                den_coeff *= sqrt_val
                den_radicand //= square
                while den_radicand % square == 0:
                    den_coeff *= sqrt_val
                    den_radicand //= square
        
        coeff = Rational(num_coeff, den_coeff)
        radicand = Rational(num_radicand, den_radicand)
        
        return (coeff, radicand)
    
    @staticmethod
    def _gcd(a: int, b: int) -> int:
        """Greatest common divisor using Euclidean algorithm."""
        while b:
            a, b = b, a % b
        return a


class CubicRoot(SymbolicExpression):
    """Represents a cube root symbolically: ?(value) + shift"""
    def __init__(self, value: Rational, shift: Rational = Rational(0, 1)):
        """
        Args:
            value: The value inside the cube root (e.g., for ?4, value=4)
            shift: Constant to add (e.g., for x = ?4 - b/(3a), shift = -b/(3a))
        """
        self.value = value
        self.shift = shift
    
    def __str__(self) -> str:
        if self.value.is_zero():
            result = "0"
        elif self.value == Rational(1, 1):
            result = "1"
        elif self.value.denominator() == 1:
            result = f"cbrt({self.value.numerator()})"
        else:
            result = f"cbrt({self.value})"
        
        if self.shift.is_zero():
            return result
        elif self.shift > Rational(0, 1):
            return f"{result} + {self.shift}"
        else:
            # shift is negative
            neg_shift = Rational(-self.shift.numerator(), self.shift.denominator())
            return f"{result} - {neg_shift}"


class SumOfCubeRoots(SymbolicExpression):
    """Represents a sum of cube roots: ?(a) + ?(b) + shift"""
    def __init__(self, value1: Rational, value2: Rational, shift: Rational = Rational(0, 1)):
        """
        Args:
            value1: First cube root value
            value2: Second cube root value
            shift: Constant to add
        """
        self.value1 = value1
        self.value2 = value2
        self.shift = shift
    
    def __str__(self) -> str:
        def format_cbrt(val: Rational) -> str:
            if val.is_zero():
                return "0"
            elif val == Rational(1, 1):
                return "1"
            elif val.denominator() == 1:
                return f"cbrt({val.numerator()})"
            else:
                return f"cbrt({val})"
        
        cbrt1 = format_cbrt(self.value1)
        cbrt2 = format_cbrt(self.value2)
        
        if self.value1.is_zero():
            result = cbrt2
        elif self.value2.is_zero():
            result = cbrt1
        else:
            result = f"{cbrt1} + {cbrt2}"
        
        if self.shift.is_zero():
            return result
        elif self.shift > Rational(0, 1):
            return f"{result} + {self.shift}"
        else:
            neg_shift = Rational(-self.shift.numerator(), self.shift.denominator())
            return f"{result} - {neg_shift}"


class NestedRadical(SymbolicExpression):
    """Represents a nested radical: ?(a + b?c) or ?(a + b?c)"""
    def __init__(self, outer_type: str, a: Rational, b: Rational, c: Rational, shift: Rational = Rational(0, 1)):
        """
        Args:
            outer_type: 'sqrt' or 'cbrt'
            a: Constant term inside radical
            b: Coefficient of inner radical
            c: Value inside inner radical
            shift: Constant to add
        """
        self.outer_type = outer_type
        self.a = a
        self.b = b
        self.c = c
        self.shift = shift
    
    def __str__(self) -> str:
        # Format inner radical: b?c
        if self.b.is_zero():
            inner = str(self.a)
        else:
            sqrt_c_coeff, sqrt_c_radicand = self._simplify_radical(self.c)
            if sqrt_c_coeff == Rational(1, 1):
                inner_sqrt = f"sqrt({sqrt_c_radicand})"
            elif sqrt_c_coeff.denominator() == 1:
                inner_sqrt = f"{sqrt_c_coeff.numerator()}*sqrt({sqrt_c_radicand})"
            else:
                inner_sqrt = f"{sqrt_c_coeff}*sqrt({sqrt_c_radicand})"
            
            if self.a.is_zero():
                inner = inner_sqrt if self.b == Rational(1, 1) else f"{self.b}*{inner_sqrt}"
            else:
                sign = "+" if self.b > Rational(0, 1) else "-"
                abs_b = abs(self.b.numerator()) if self.b.denominator() == 1 else abs(self.b)
                if abs_b == Rational(1, 1):
                    inner = f"{self.a} {sign} {inner_sqrt}"
                else:
                    inner = f"{self.a} {sign} {abs_b}*{inner_sqrt}"
        
        # Format outer radical
        if self.outer_type == 'sqrt':
            result = f"sqrt({inner})"
        else:  # cbrt
            result = f"cbrt({inner})"
        
        if self.shift.is_zero():
            return result
        elif self.shift > Rational(0, 1):
            return f"{result} + {self.shift}"
        else:
            neg_shift = Rational(-self.shift.numerator(), self.shift.denominator())
            return f"{result} - {neg_shift}"
    
    def _simplify_radical(self, r: Rational) -> Tuple[Rational, Rational]:
        """Helper to simplify radicals."""
        if r < Rational(0, 1):
            return (Rational(1, 1), r)
        
        if r.is_zero():
            return (Rational(0, 1), Rational(0, 1))
        
        num = abs(r.numerator())
        den = abs(r.denominator())
        
        num_coeff = 1
        num_radicand = num
        
        for square in [4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225]:
            if num_radicand % square == 0:
                sqrt_val = int(square ** 0.5)
                num_coeff *= sqrt_val
                num_radicand //= square
                while num_radicand % square == 0:
                    num_coeff *= sqrt_val
                    num_radicand //= square
        
        den_coeff = 1
        den_radicand = den
        
        for square in [4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225]:
            if den_radicand % square == 0:
                sqrt_val = int(square ** 0.5)
                den_coeff *= sqrt_val
                den_radicand //= square
                while den_radicand % square == 0:
                    den_coeff *= sqrt_val
                    den_radicand //= square
        
        coeff = Rational(num_coeff, den_coeff)
        radicand = Rational(num_radicand, den_radicand)
        
        return (coeff, radicand)


class SymbolicSolver:
    """Symbolic equation solver for polynomials and general equations."""
    
    def solve(self, coeffs: List[Rational], var: str) -> List[Solution]:
        """
        Solve polynomial equation symbolically.
        
        This is the main entry point for solving equations.
        
        Args:
            coeffs: List of coefficients [a0, a1, ..., an] for a0 + a1*x + ... + an*x^n = 0
            var: Variable name
        
        Returns:
            List of symbolic solutions
        """
        return self.solve_polynomial(coeffs, var)
    
    def solve_polynomial(self, coeffs: List[Rational], var: str) -> List[Solution]:
        """
        Solve polynomial equation symbolically.
        
        Args:
            coeffs: List of coefficients [a0, a1, ..., an] for a0 + a1*x + ... + an*x^n = 0
            var: Variable name
        
        Returns:
            List of symbolic solutions
        """
        # Remove leading zeros
        coeffs = coeffs[:]  # Make a copy to avoid modifying input
        while coeffs and coeffs[-1].is_zero():
            coeffs.pop()
        
        if not coeffs:
            # Zero polynomial - infinite solutions
            return [Solution.all_reals(var)]
        
        degree = len(coeffs) - 1
        
        # Handle degenerate cases
        if degree == 0:
            # Constant: c = 0
            if coeffs[0].is_zero():
                return [Solution.all_reals(var)]
            else:
                return [Solution.no_solution(var)]
        
        if degree == 1:
            return self.solve_linear(coeffs, var)
        
        if degree == 2:
            return self.solve_quadratic(coeffs, var)
        
        # For higher degrees, try to find rational roots first
        # Rational Root Theorem: if p/q is a root, then p divides constant and q divides leading coefficient
        rational_roots = self.find_rational_roots(coeffs)

        if rational_roots:
            solutions = []
            remaining_coeffs = coeffs[:]
            seen_roots = set()  # Track which roots we've already processed

            for root in rational_roots:
                # Skip if we've already processed this root
                root_tuple = (root.numerator(), root.denominator())  # Use tuple for hashing
                if root_tuple in seen_roots:
                    continue
                
                # Count how many times this root divides the polynomial
                multiplicity = 0
                test_coeffs = remaining_coeffs[:]
                
                # Keep dividing by (x - root) until it no longer divides
                while len(test_coeffs) > 1:
                    # Evaluate polynomial at root to check if it's zero
                    eval_result = self.evaluate_polynomial(test_coeffs, root)
                    if eval_result.is_zero():
                        multiplicity += 1
                        test_coeffs = self.synthetic_divide(test_coeffs, root)
                    else:
                        break
                
                if multiplicity > 0:
                    seen_roots.add(root_tuple)
                    solutions.append(Solution(var, root, multiplicity=multiplicity))
                    
                    # Factor out all occurrences of (x - root)
                    for _ in range(multiplicity):
                        remaining_coeffs = self.synthetic_divide(remaining_coeffs, root)

            # Recursively solve the reduced polynomial
            if len(remaining_coeffs) > 1:
                remaining_solutions = self.solve_polynomial(remaining_coeffs, var)
                # Filter out solutions we've already found
                for sol in remaining_solutions:
                    if isinstance(sol.value, Rational):
                        sol_tuple = (sol.value.numerator(), sol.value.denominator())
                        if sol_tuple not in seen_roots:
                            solutions.append(sol)
                    else:
                        # For symbolic solutions, add them (they're different)
                        solutions.append(sol)

            return solutions

        # No rational roots found
        # For degree 3, could use Cardano's method
        # For degree 4, could use Ferrari's method
        # For degree >= 5, no general closed-form solution exists

        if degree == 3:
            return self.solve_cubic(coeffs, var)

        if degree == 4:
            return self.solve_quartic(coeffs, var)

        # Degree >= 5: no closed-form solution
        # Could return symbolic representation or fall back to numerical
        return []

    def solve_linear(self, coeffs: List[Rational], var: str) -> List[Solution]:
        """
        Solve linear equation: ax + b = 0 ? x = -b/a
        
        Args:
            coeffs: List of coefficients [b, a] where equation is a*x + b = 0
            var: Variable name
        
        Returns:
            List of solutions
        """
        if len(coeffs) < 1:
            return []
        
        if len(coeffs) == 1:
            # Just constant: b = 0
            if coeffs[0].is_zero():
                return [Solution.all_reals(var)]  # Infinite solutions
            else:
                return [Solution.no_solution(var)]  # No solution
        
        a = coeffs[1]  # coefficient of x
        b = coeffs[0]  # constant term
        
        if a.is_zero():
            # 0*x + b = 0
            if b.is_zero():
                return [Solution.all_reals(var)]
            else:
                return [Solution.no_solution(var)]
        
        # x = -b/a
        solution_value = Rational(-b.numerator(), b.denominator()) / a
        return [Solution(var, solution_value)]
    
    def solve_quadratic(self, coeffs: List[Rational], var: str) -> List[Solution]:
        """
        Solve quadratic equation: ax? + bx + c = 0
        
        Uses quadratic formula: x = (-b ? ?(b?-4ac))/(2a)
        
        Args:
            coeffs: List of coefficients [c, b, a] where equation is a*x? + b*x + c = 0
            var: Variable name
        
        Returns:
            List of symbolic solutions
        """
        if len(coeffs) < 2:
            # Degenerate to linear or constant
            return self.solve_linear(coeffs, var)
        
        if len(coeffs) == 2:
            # Linear case: bx + c = 0
            return self.solve_linear(coeffs, var)
        
        a = coeffs[2]
        b = coeffs[1]
        c = coeffs[0]
        
        if a.is_zero():
            # Degenerate to linear
            return self.solve_linear(coeffs[:2], var)
        
        # Discriminant: ? = b? - 4ac
        discriminant = b * b - Rational(4, 1) * a * c
        
        if discriminant.is_zero():
            # One repeated root: x = -b/(2a)
            root = Rational(-b.numerator(), b.denominator()) / (Rational(2, 1) * a)
            return [Solution(var, root, multiplicity=2)]
        
        # Check if discriminant is a perfect square
        sqrt_disc = self.sqrt_rational(discriminant)
        
        if sqrt_disc is not None:
            # Exact square root exists - return exact solutions
            two_a = Rational(2, 1) * a
            neg_b = Rational(-b.numerator(), b.denominator())
            
            root1 = (neg_b + sqrt_disc) / two_a
            root2 = (neg_b - sqrt_disc) / two_a
            return [Solution(var, root1), Solution(var, root2)]
        
        elif discriminant > Rational(0, 1):
            # Positive discriminant but not perfect square - return symbolic form
            return [
                Solution(var, QuadraticRoot(a, b, c, positive=True)),
                Solution(var, QuadraticRoot(a, b, c, positive=False))
            ]
        else:
            # Negative discriminant - complex roots
            # For now, return empty (TODO: implement complex number support)
            return []

    def solve_cubic(self, coeffs: List[Rational], var: str) -> List[Solution]:
        """
        Solve cubic equation: ax? + bx? + cx + d = 0 using Cardano's method.
        
        Args:
            coeffs: List of coefficients [d, c, b, a] where equation is a*x? + b*x? + c*x + d = 0
            var: Variable name
        
        Returns:
            List of symbolic solutions
        """
        if len(coeffs) < 4:
            return self.solve_quadratic(coeffs, var)
        
        a = coeffs[3]
        b = coeffs[2]
        c = coeffs[1]
        d = coeffs[0]
        
        if a.is_zero():
            return self.solve_quadratic(coeffs[:3], var)
        
        # Try rational roots first (most common case)
        rational_roots = self.find_rational_roots(coeffs)
        if rational_roots:
            solutions = []
            remaining_coeffs = coeffs[:]
            for root in rational_roots:
                solutions.append(Solution(var, root))
                remaining_coeffs = self.synthetic_divide(remaining_coeffs, root)
            if len(remaining_coeffs) > 1:
                solutions.extend(self.solve_polynomial(remaining_coeffs, var))
            return solutions
        
        # Normalize to monic: x? + (b/a)x? + (c/a)x + (d/a) = 0
        # But we'll work with the depressed cubic form instead
        
        # Convert to depressed cubic: t? + pt + q = 0 where x = t - b/(3a)
        # p = (3ac - b?) / (3a?)
        # q = (2b? - 9abc + 27a?d) / (27a?)
        
        three_a = Rational(3, 1) * a
        three_a_sq = three_a * a
        twenty_seven_a_cubed = Rational(27, 1) * a * a * a
        
        p = (three_a * c - b * b) / three_a_sq
        q = (Rational(2, 1) * b * b * b - Rational(9, 1) * a * b * c + Rational(27, 1) * a * a * d) / twenty_seven_a_cubed
        
        # Discriminant: ? = (q/2)? + (p/3)?
        q_half = q / Rational(2, 1)
        p_third = p / Rational(3, 1)
        discriminant = q_half * q_half + p_third * p_third * p_third
        
        # Shift back: x = t - b/(3a)
        shift = b / (Rational(3, 1) * a)
        
        if discriminant > Rational(0, 1):
            # One real root, two complex
            # Cardano's formula: t = ?(-q/2 + ??) + ?(-q/2 - ??)
            # But if p=0, this simplifies to t = ?(-q)
            if p.is_zero():
                # Simple case: t? + q = 0, so t = ?(-q)
                t = CubicRoot(-q, Rational(0, 1))
                x = CubicRoot(-q, -shift)
                return [Solution(var, x)]
            else:
                # General case: use Cardano's formula
                # t = ?(-q/2 + ??) + ?(-q/2 - ??)
                sqrt_disc = self.sqrt_rational(discriminant)
                if sqrt_disc is not None:
                    # Discriminant is a perfect square
                    term1 = -q_half + sqrt_disc
                    term2 = -q_half - sqrt_disc
                    # For the real root, we use the principal cube root
                    # t = ?(term1) + ?(term2)
                    # Check if terms are perfect cubes
                    cbrt1 = self.cube_root_rational(term1)
                    cbrt2 = self.cube_root_rational(term2)
                    if cbrt1 is not None and cbrt2 is not None:
                        # Both cube roots are rational
                        t = cbrt1 + cbrt2
                        x = t - shift
                        return [Solution(var, x)]
                    else:
                        # Return symbolic representation using SumOfCubeRoots
                        # Cardano's formula: t = ?(term1) + ?(term2)
                        return [Solution(var, SumOfCubeRoots(term1, term2, -shift))]
                else:
                    # Discriminant is not a perfect square - nested radicals
                    # Cardano's formula: t = ?(-q/2 + ??) + ?(-q/2 - ??)
                    # This requires representing nested radicals (?? inside cube root)
                    # For now, represent as SumOfCubeRoots with the terms
                    # TODO: Implement proper nested radical representation (item 5)
                    term1 = -q_half
                    term2 = -q_half  # Will be handled as nested radical later
                    return [Solution(var, SumOfCubeRoots(term1, term2, -shift))]
        elif discriminant.is_zero():
            # All roots real, at least two equal
            if p.is_zero() and q.is_zero():
                # Triple root at t = 0
                root = -shift
                return [Solution(var, root, multiplicity=3)]
            else:
                # Double root and single root
                # t? = 2 * cube_root(-q/2)
                # t? = t? = -cube_root(-q/2)
                neg_q_half = -q_half
                cube_root_val = self.cube_root_rational(neg_q_half)
                if cube_root_val is not None:
                    t1 = Rational(2, 1) * cube_root_val
                    t2 = -cube_root_val
                    root1 = t1 - shift
                    root2 = t2 - shift
                    return [
                        Solution(var, root1),
                        Solution(var, root2, multiplicity=2)
                    ]
        
        else:
            # Discriminant < 0: All three roots are real and distinct
            # Use trigonometric solution or Cardano's formula with complex cube roots
            # For now, try to return at least one real root symbolically
            # The three real roots are:
            # t_k = 2 * sqrt(-p/3) * cos((1/3) * arccos(-q/2 * sqrt(-27/p?)) - 2?k/3) for k=0,1,2
            # This is complex to represent symbolically, so for now we return empty
            # TODO: Implement trigonometric method or return all three roots symbolically
            return []
    
    def cube_root_rational(self, r: Rational) -> Optional[Rational]:
        """
        Check if ?r simplifies to a rational number.
        Returns the rational cube root if it exists, None otherwise.
        """
        if r.is_zero():
            return Rational(0, 1)
        
        # Check if numerator and denominator are perfect cubes
        num = abs(r.numerator())
        den = abs(r.denominator())
        sign = -1 if (r.numerator() < 0) != (r.denominator() < 0) else 1
        
        # Find integer cube roots
        num_cbrt = int(round(num ** (1/3)))
        den_cbrt = int(round(den ** (1/3)))
        
        if num_cbrt * num_cbrt * num_cbrt == num and den_cbrt * den_cbrt * den_cbrt == den:
            return Rational(sign * num_cbrt, den_cbrt)
        
        return None
    
    def solve_quartic(self, coeffs: List[Rational], var: str) -> List[Solution]:
        """
        Solve quartic equation: ax? + bx? + cx? + dx + e = 0 using Ferrari's method.
        
        For exact solutions, tries rational roots first. If none found,
        converts to depressed quartic and solves resolvent cubic.
        
        Args:
            coeffs: List of coefficients [e, d, c, b, a] where equation is a*x? + b*x? + c*x? + d*x + e = 0
            var: Variable name
        
        Returns:
            List of symbolic solutions
        """
        if len(coeffs) < 5:
            return self.solve_cubic(coeffs, var)
        
        a = coeffs[4]
        b = coeffs[3]
        c = coeffs[2]
        d = coeffs[1]
        e = coeffs[0]
        
        if a.is_zero():
            return self.solve_cubic(coeffs[:4], var)
        
        # Try rational roots first (most common case)
        rational_roots = self.find_rational_roots(coeffs)
        if rational_roots:
            solutions = []
            remaining_coeffs = coeffs[:]
            for root in rational_roots:
                solutions.append(Solution(var, root))
                remaining_coeffs = self.synthetic_divide(remaining_coeffs, root)
            if len(remaining_coeffs) > 1:
                solutions.extend(self.solve_polynomial(remaining_coeffs, var))
            return solutions
        
        # Check for biquadratic form: ax? + cx? + e = 0 (no x? or x terms)
        if b.is_zero() and d.is_zero():
            # Substitute u = x? to get: au? + cu + e = 0
            # Solve quadratic in u, then take square roots
            quad_coeffs = [e, c, a]  # [e, c, a] for au? + cu + e = 0
            u_solutions = self.solve_quadratic(quad_coeffs, var)
            
            solutions = []
            for u_sol in u_solutions:
                if isinstance(u_sol.value, Rational):
                    u_val = u_sol.value
                    if u_val >= Rational(0, 1):
                        # u = x?, so x = ??u
                        sqrt_u = self.sqrt_rational(u_val)
                        if sqrt_u is not None:
                            # Exact square root
                            solutions.append(Solution(var, sqrt_u))
                            if not sqrt_u.is_zero():
                                solutions.append(Solution(var, -sqrt_u))
                        else:
                            # Symbolic square root
                            solutions.append(Solution(var, QuadraticRoot(Rational(1, 1), Rational(0, 1), -u_val, positive=True)))
                            solutions.append(Solution(var, QuadraticRoot(Rational(1, 1), Rational(0, 1), -u_val, positive=False)))
                elif isinstance(u_sol.value, QuadraticRoot):
                    # u is a quadratic root, so x = ??u
                    # This is complex - would need nested radicals
                    # For now, skip (TODO: implement nested radical handling)
                    pass
            
            if solutions:
                return solutions
        
        # Check for quartic of the form x? - c = 0 (simple case)
        if b.is_zero() and c.is_zero() and d.is_zero():
            # x? = -e/a, so x? = constant
            constant = -e / a
            if constant >= Rational(0, 1):
                # x? = positive constant
                # Factor as (x? - ?constant)(x? + ?constant) = 0
                sqrt_const = self.sqrt_rational(constant)
                if sqrt_const is not None:
                    # constant is a perfect square
                    # x? = ?constant or x? = -?constant
                    # Real solutions: x = ??(?constant)
                    sqrt_sqrt_const = self.sqrt_rational(sqrt_const)
                    if sqrt_sqrt_const is not None:
                        # Fourth root is rational
                        solutions = [
                            Solution(var, sqrt_sqrt_const),
                            Solution(var, -sqrt_sqrt_const)
                        ]
                        return solutions
                    else:
                        # x = ??(?constant) where ?constant is rational but ?(?constant) is not
                        # x? = ?constant, so x = ??(?constant)
                        # This requires nested square roots
                        # For now, represent as x = ??(?constant)
                        # Actually, we can use: x? = ?constant, so x = ??(?constant)
                        # But we don't have nested radical support yet
                        # Workaround: solve x? - ?constant = 0
                        return self.solve_quadratic([-sqrt_const, Rational(0, 1), Rational(1, 1)], var)
                else:
                    # constant is not a perfect square
                    # x? = constant, so we can solve as biquadratic
                    # This should have been caught above, but handle it here too
                    quad_coeffs = [e, Rational(0, 1), a]
                    u_solutions = self.solve_quadratic(quad_coeffs, var)
                    solutions = []
                    for u_sol in u_solutions:
                        if isinstance(u_sol.value, Rational):
                            u_val = u_sol.value
                            if u_val >= Rational(0, 1):
                                sqrt_u = self.sqrt_rational(u_val)
                                if sqrt_u is not None:
                                    solutions.append(Solution(var, sqrt_u))
                                    if not sqrt_u.is_zero():
                                        solutions.append(Solution(var, -sqrt_u))
                                else:
                                    solutions.append(Solution(var, QuadraticRoot(Rational(1, 1), Rational(0, 1), -u_val, positive=True)))
                                    solutions.append(Solution(var, QuadraticRoot(Rational(1, 1), Rational(0, 1), -u_val, positive=False)))
                    if solutions:
                        return solutions
        
        # Ferrari's method: Convert to depressed quartic and solve resolvent cubic
        # Convert to depressed quartic: t? + pt? + qt + r = 0 where x = t - b/(4a)
        # Shift: x = t - b/(4a)
        shift = b / (Rational(4, 1) * a)
        
        # Compute depressed quartic coefficients
        # p = (8ac - 3b?) / (8a?)
        # q = (b? - 4abc + 8a?d) / (8a?)
        # r = (-3b? + 16ab?c - 64a?bd + 256a?e) / (256a?)
        
        eight_a = Rational(8, 1) * a
        eight_a_sq = eight_a * a
        eight_a_cubed = eight_a_sq * a
        two_fifty_six_a_fourth = Rational(256, 1) * a * a * a * a
        
        p = (eight_a * c - Rational(3, 1) * b * b) / eight_a_sq
        q = (b * b * b - Rational(4, 1) * a * b * c + eight_a_sq * d) / eight_a_cubed
        r = (Rational(-3, 1) * b * b * b * b + Rational(16, 1) * a * b * b * c - 
             Rational(64, 1) * a * a * b * d + Rational(256, 1) * a * a * a * e) / two_fifty_six_a_fourth
        
        # Resolvent cubic: u? + 2pu? + (p? - 4r)u - q? = 0
        # Solve for u, then use it to factor the depressed quartic
        resolvent_coeffs = [
            -q * q,  # constant term
            p * p - Rational(4, 1) * r,  # u coefficient
            Rational(2, 1) * p,  # u? coefficient
            Rational(1, 1)  # u? coefficient
        ]
        
        u_solutions = self.solve_cubic(resolvent_coeffs, var)
        
        if not u_solutions:
            # Couldn't solve resolvent cubic, return empty
            return []
        
        # Find a real root u (prefer rational if available)
        u_val = None
        for u_sol in u_solutions:
            if isinstance(u_sol.value, Rational):
                u_val = u_sol.value
                break
        
        if u_val is None:
            # No rational root found, try to use first solution symbolically
            # For now, return empty (would need symbolic manipulation)
            return []
        
        if u_val < Rational(0, 1):
            # u must be non-negative for Ferrari's method
            # Try next solution or return empty
            return []
        
        # Factor depressed quartic using u
        # t? + pt? + qt + r = (t? + ?t + ?)(t? - ?t + ?)
        # where ?? = 2u, ? and ? are determined from p, q, r, u
        
        # ? = ?(2u)
        alpha_sq = Rational(2, 1) * u_val
        sqrt_alpha_sq = self.sqrt_rational(alpha_sq)
        
        if sqrt_alpha_sq is None:
            # ? is not rational, would need symbolic sqrt
            # For now, return empty
            return []
        
        alpha = sqrt_alpha_sq
        
        # ? and ? from: ? + ? = p + ??, ? - ? = q/?
        beta_plus_gamma = p + alpha_sq
        beta_minus_gamma = q / alpha if not alpha.is_zero() else Rational(0, 1)
        
        beta = (beta_plus_gamma + beta_minus_gamma) / Rational(2, 1)
        gamma = (beta_plus_gamma - beta_minus_gamma) / Rational(2, 1)
        
        # Now solve the two quadratics:
        # t? + ?t + ? = 0 and t? - ?t + ? = 0
        solutions = []
        
        # First quadratic: t? + ?t + ? = 0
        quad1_coeffs = [beta, alpha, Rational(1, 1)]
        quad1_sols = self.solve_quadratic(quad1_coeffs, var)
        for sol in quad1_sols:
            if isinstance(sol.value, Rational):
                x_val = sol.value - shift
                solutions.append(Solution(var, x_val))
            elif isinstance(sol.value, QuadraticRoot):
                # Create new QuadraticRoot with shifted variable
                # x = t - shift, so we need to adjust
                # Actually, QuadraticRoot represents the solution in terms of t
                # We need to subtract shift
                # For now, add as-is (would need to modify QuadraticRoot to support shift)
                solutions.append(sol)  # TODO: properly handle shift
        
        # Second quadratic: t? - ?t + ? = 0
        quad2_coeffs = [gamma, -alpha, Rational(1, 1)]
        quad2_sols = self.solve_quadratic(quad2_coeffs, var)
        for sol in quad2_sols:
            if isinstance(sol.value, Rational):
                x_val = sol.value - shift
                solutions.append(Solution(var, x_val))
            elif isinstance(sol.value, QuadraticRoot):
                solutions.append(sol)  # TODO: properly handle shift
        
        return solutions
    
    def sqrt_rational(self, r: Rational) -> Optional[Rational]:
        """
        Check if ?r simplifies to a rational number.
        Returns the rational square root if it exists, None otherwise.
        
        Only works for perfect squares: ?(n?/d?) = n/d
        """
        if r < Rational(0, 1):
            return None  # Negative, would be complex
        
        # Check if numerator and denominator are perfect squares
        num = r.numerator()
        den = r.denominator()
        
        # Find integer square roots
        num_sqrt = int(num ** 0.5)
        den_sqrt = int(den ** 0.5)
        
        if num_sqrt * num_sqrt == num and den_sqrt * den_sqrt == den:
            return Rational(num_sqrt, den_sqrt)
        
        return None
    
    def simplify_radical(self, r: Rational) -> Tuple[Rational, Rational]:
        """
        Factor perfect squares from a radical: ?(a*b?) = b?a
        
        Returns (coeff, radicand) where ?r = coeff * ?radicand
        and radicand is square-free (or as close as possible).
        
        Example: simplify_radical(8) ? (2, 2) because ?8 = 2?2
        Example: simplify_radical(12) ? (2, 3) because ?12 = 2?3
        """
        if r < Rational(0, 1):
            return (Rational(0, 1), r)  # Can't simplify negative
        
        if r.is_zero():
            return (Rational(0, 1), Rational(0, 1))
        
        num = abs(r.numerator())
        den = abs(r.denominator())
        sign = -1 if (r.numerator() < 0) != (r.denominator() < 0) else 1
        
        # Factor perfect squares from numerator
        num_coeff = Rational(1, 1)
        num_radicand = num
        
        # Try to extract perfect square factors
        # Simple approach: check if divisible by small perfect squares
        for square in [4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225]:
            if num_radicand % square == 0:
                sqrt_val = int(square ** 0.5)
                num_coeff = num_coeff * Rational(sqrt_val, 1)
                num_radicand = num_radicand // square
                # Check again (e.g., 72 = 36*2, then 2*36 = 72 again)
                while num_radicand % square == 0:
                    num_coeff = num_coeff * Rational(sqrt_val, 1)
                    num_radicand = num_radicand // square
        
        # Factor perfect squares from denominator
        den_coeff = Rational(1, 1)
        den_radicand = den
        
        for square in [4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225]:
            if den_radicand % square == 0:
                sqrt_val = int(square ** 0.5)
                den_coeff = den_coeff * Rational(sqrt_val, 1)
                den_radicand = den_radicand // square
                while den_radicand % square == 0:
                    den_coeff = den_coeff * Rational(sqrt_val, 1)
                    den_radicand = den_radicand // square
        
        # Combine: ?(num/den) = (num_coeff/den_coeff) * ?(num_radicand/den_radicand)
        coeff = num_coeff / den_coeff
        radicand = Rational(num_radicand, den_radicand)
        
        return (coeff, radicand)
    
    def find_rational_roots(self, coeffs: List[Rational]) -> List[Rational]:
        """
        Find rational roots using Rational Root Theorem.
        
        If p/q is a root, then:
        - p divides the constant term (a0)
        - q divides the leading coefficient (an)
        
        Returns list of rational roots found.
        """
        if len(coeffs) < 2:
            return []
        
        # Get constant term and leading coefficient
        constant = coeffs[0]
        leading = coeffs[-1]
        
        if constant.is_zero():
            # 0 is a root
            return [Rational(0, 1)]
        
        # Find divisors of numerator and denominator
        const_num = abs(constant.numerator())
        const_den = abs(constant.denominator())
        lead_num = abs(leading.numerator())
        lead_den = abs(leading.denominator())
        
        # Generate candidate roots: ?(p/q) where p|const_num, q|lead_den
        candidates = []
        
        # Simple approach: try small integers first
        for p in range(1, min(const_num + 1, 100)):  # Limit search
            if const_num % p == 0:
                for q in range(1, min(lead_den + 1, 100)):
                    if lead_den % q == 0:
                        # Try ?p/q
                        candidate_pos = Rational(p, q)
                        candidate_neg = Rational(-p, q)
                        candidates.append(candidate_pos)
                        candidates.append(candidate_neg)
        
        # Test candidates
        roots = []
        for candidate in candidates:
            if self.evaluate_polynomial(coeffs, candidate).is_zero():
                roots.append(candidate)
        
        return roots
    
    def evaluate_polynomial(self, coeffs: List[Rational], x: Rational) -> Rational:
        """Evaluate polynomial at x using Horner's method."""
        result = Rational(0, 1)
        for i in range(len(coeffs) - 1, -1, -1):
            result = result * x + coeffs[i]
        return result
    
    def synthetic_divide(self, coeffs: List[Rational], root: Rational) -> List[Rational]:
        """
        Divide polynomial by (x - root) using synthetic division.
        
        Returns coefficients of quotient polynomial.
        """
        if len(coeffs) < 2:
            return []
        
        # Horner's method for synthetic division
        result = []
        carry = Rational(0, 1)
        
        # Process coefficients from highest degree to lowest
        for i in range(len(coeffs) - 1, 0, -1):
            carry = coeffs[i] + carry * root
            result.append(carry)
        
        # Reverse to get standard form [a0, a1, ..., an]
        result.reverse()
        return result


# Create a default instance for backward compatibility
_default_solver = SymbolicSolver()

# Backward compatibility functions
def solve_polynomial_symbolic(coeffs: List[Rational], var: str) -> List[Solution]:
    """Backward compatibility wrapper."""
    return _default_solver.solve_polynomial(coeffs, var)

def solve_linear(coeffs: List[Rational], var: str) -> List[Solution]:
    """Backward compatibility wrapper."""
    return _default_solver.solve_linear(coeffs, var)

def solve_quadratic(coeffs: List[Rational], var: str) -> List[Solution]:
    """Backward compatibility wrapper."""
    return _default_solver.solve_quadratic(coeffs, var)

def solve_cubic(coeffs: List[Rational], var: str) -> List[Solution]:
    return _default_solver.solve_cubic(coeffs, var)

def solve_quartic(coeffs: List[Rational], var: str) -> List[Solution]:
    return _default_solver.solve_quartic(coeffs, var)

def find_rational_roots(coeffs: List[Rational]) -> List[Rational]:
    """Backward compatibility wrapper."""
    return _default_solver.find_rational_roots(coeffs)
