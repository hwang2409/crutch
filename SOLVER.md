# Symbolic Solver: Complete Guide & Implementation

## Overview

A symbolic solver takes an equation like `x² - 4 = 0` and returns exact symbolic solutions like `x = 2` or `x = -2`, rather than numerical approximations.

---

## What's Been Built

I've created a **working symbolic solver** that integrates with your existing CAS. Here's what you now have:

### ✓ **Working Features**

1. **Linear Equations**: `2x + 4 = 0` → `x = -2`
2. **Quadratic Equations with Exact Roots**: `x² - 4 = 0` → `x = 2, x = -2`
3. **Quadratic Equations with Symbolic Roots**: `x² - 2 = 0` → `x = ±√2` (symbolic form)
4. **Polynomial Root Finding**: Uses Rational Root Theorem to find rational roots
5. **Synthetic Division**: Factors out found roots and recursively solves

### 📁 **Files Created**

1. **`solver.py`** - Core solver implementation
   - `Solution` class - represents solutions
   - `solve_linear()` - solves linear equations
   - `solve_quadratic()` - solves quadratic equations  
   - `solve_polynomial_symbolic()` - general polynomial solver
   - `find_rational_roots()` - Rational Root Theorem implementation
   - Helper functions for polynomial manipulation

2. **`SOLVER.md`** - This comprehensive guide
   - Architecture overview
   - Step-by-step approach for each equation type
   - Implementation strategies
   - Testing approach

3. **`solver_examples.py`** - Demonstration and testing

### 🔌 **Integration**

The solver is integrated into your CAS:
```python
cas = CAS()
expr = cas.parse("x^2 - 4")
solutions = expr.solve("x")  # Returns [Solution('x', 2), Solution('x', -2)]
```

Or via the CAS method:
```python
solutions = cas.solve("x^2 - 4", "x")
```

---

## Architecture

The solver should:
1. **Parse equations** - Handle `expr = 0` or `expr1 = expr2` format
2. **Normalize** - Convert to `expr = 0` form
3. **Simplify** - Reduce to simplest form
4. **Classify** - Determine equation type (linear, quadratic, polynomial, etc.)
5. **Solve** - Apply appropriate solving method
6. **Return** - Symbolic solutions (not numerical approximations)

### **Architecture Flow**

```
User Input: "x^2 - 4"
    →
CAS.parse() → ExprResult (Polynomial)
    →
expr.solve("x")
    →
solve_polynomial_symbolic(coeffs, "x")
    →
Classify by degree:
  - Degree 1 → solve_linear()
  - Degree 2 → solve_quadratic()
  - Degree 3+ → find_rational_roots() → synthetic_divide() → recurse
    →
Return List[Solution]
```

### **Key Algorithms**

1. **Linear Solver**: Direct formula `x = -b/a`
2. **Quadratic Solver**: Quadratic formula with perfect square detection
3. **Rational Root Theorem**: Tests candidate roots `√p/q` where:
   - `p` divides constant term
   - `q` divides leading coefficient
4. **Synthetic Division**: Efficiently factors out roots

---

## Step-by-Step Implementation Guide

### Step 1: Linear Equations (ax + b = 0)

**Simplest case**: `ax + b = 0` → `x = -b/a`

#### Implementation Strategy

```python
def solve_linear(coeffs: list[Rational], var: str) -> list["Solution"]:
    """
    Solve ax + b = 0 where coeffs = [b, a] (constant, linear)
    Returns: [Solution(var, -b/a)]
    """
    if len(coeffs) < 2:
        if len(coeffs) == 1:
            # Just constant: b = 0
            if coeffs[0].is_zero():
                return [Solution.all_reals(var)]  # Infinite solutions
            else:
                return []  # No solution
        return []
    
    a = coeffs[1]  # coefficient of x
    b = coeffs[0]  # constant term
    
    if a.is_zero():
        # 0*x + b = 0
        if b.is_zero():
            return [Solution.all_reals(var)]
        else:
            return []  # No solution
    
    # x = -b/a
    solution = Rational(-b.numerator(), b.denominator()) / a
    return [Solution(var, solution)]
```

**Key insight**: For linear equations, you can extract coefficients directly from the polynomial representation.

---

### Step 2: Quadratic Equations (ax² + bx + c = 0)

**Standard form**: `ax² + bx + c = 0` → `x = (-b → √(b²-4ac)) / 2a`

#### Implementation Strategy

```python
def solve_quadratic(coeffs: list[Rational], var: str) -> list["Solution"]:
    """
    Solve ax² + bx + c = 0 where coeffs = [c, b, a]
    Returns: List of symbolic solutions
    """
    if len(coeffs) < 3:
        # Degenerate to linear
        return solve_linear(coeffs, var)
    
    a = coeffs[2]
    b = coeffs[1]
    c = coeffs[0]
    
    if a.is_zero():
        return solve_linear(coeffs[:2], var)
    
    # Discriminant: → = b² - 4ac
    discriminant = b * b - Rational(4, 1) * a * c
    
    if discriminant.is_zero():
        # One repeated root: x = -b/(2a)
        root = Rational(-b.numerator(), b.denominator()) / (Rational(2, 1) * a)
        return [Solution(var, root)]
    
    elif discriminant > Rational(0, 1):
        # Two real roots
        sqrt_disc = sqrt_rational(discriminant)  # Need to implement
        if sqrt_disc is None:
            # Can't simplify sqrt, return symbolic form
            return [
                Solution(var, QuadraticRoot(a, b, c, positive=True)),
                Solution(var, QuadraticRoot(a, b, c, positive=False))
            ]
        else:
            # Exact square root
            root1 = (Rational(-b.numerator(), b.denominator()) + sqrt_disc) / (Rational(2, 1) * a)
            root2 = (Rational(-b.numerator(), b.denominator()) - sqrt_disc) / (Rational(2, 1) * a)
            return [Solution(var, root1), Solution(var, root2)]
    
    else:
        # Complex roots (for now, return empty or symbolic complex)
        # TODO: Implement complex number support
        return []
```

**Challenges**:
- Need to handle square roots symbolically
- May need to return `(-b → ±)/(2a)` as symbolic expression
- Complex roots require complex number support

---

### Step 3: Cubic Equations (ax² + bx² + cx + d = 0)

**Cardano's formula**: More complex, but still algebraic.

#### Strategy
- Use Cardano's method for cubic equations
- Returns symbolic roots (may involve cube roots)
- Falls back to numerical for irreducible cases

---

### Step 4: Higher-Degree Polynomials

#### Strategy 1: Factorization
Try to factor the polynomial:
- Look for integer roots (rational root theorem)
- Factor out common factors
- Factor by grouping
- Use known factorizations

#### Strategy 2: Radical Solutions
- For degree 4: Use Ferrari's method
- For degree → 5: May not have closed-form solutions (Abel-Ruffini theorem)
- Fall back to numerical or symbolic representation

#### Implementation

```python
def solve_polynomial(coeffs: list[Rational], var: str) -> list["Solution"]:
    """
    Solve polynomial equation of arbitrary degree.
    Tries factorization first, then closed-form if possible.
    """
    degree = len(coeffs) - 1
    
    # Degenerate cases
    if degree == 0:
        return solve_constant(coeffs[0])
    if degree == 1:
        return solve_linear(coeffs, var)
    if degree == 2:
        return solve_quadratic(coeffs, var)
    if degree == 3:
        return solve_cubic(coeffs, var)
    if degree == 4:
        return solve_quartic(coeffs, var)
    
    # Try factorization
    factors = factor_polynomial(coeffs)
    if len(factors) > 1:
        # Recursively solve each factor
        solutions = []
        for factor in factors:
            solutions.extend(solve_polynomial(factor, var))
        return solutions
    
    # Try rational root theorem
    rational_roots = find_rational_roots(coeffs)
    if rational_roots:
        # Factor out (x - root) and recurse
        solutions = []
        remaining = coeffs
        for root in rational_roots:
            solutions.append(Solution(var, root))
            remaining = polynomial_divide(remaining, root)
        if len(remaining) > 1:
            solutions.extend(solve_polynomial(remaining, var))
        return solutions
    
    # No closed-form solution possible
    # Return symbolic representation or fall back to numerical
    return [Solution.symbolic(var, f"Root of polynomial: {polynomial_to_string(coeffs)}")]
```

---

### Step 5: General Equations (Non-Polynomial)

For equations like `sin(x) = 0` or `exp(x) = 2`:

#### Strategy: Pattern Matching

```python
def solve_general(expr: EDAG, var: str) -> list["Solution"]:
    """
    Solve general equations using pattern matching.
    """
    # Try to recognize patterns
    
    # Pattern: sin(f(x)) = 0 → f(x) = n√     if matches_pattern(expr, "sin(...) = 0"):
        inner = extract_inner(expr)
        return solve(inner, var)  # Recursive
    
    # Pattern: exp(f(x)) = c → f(x) = ln(c)
    if matches_pattern(expr, "exp(...) = c"):
        inner = extract_inner(expr)
        c = extract_constant(expr)
        return solve(inner == log(c), var)
    
    # Pattern: f(x)^n = c → f(x) = c^(1/n)
    if matches_pattern(expr, "...^n = c"):
        # Extract base and exponent
        # Solve recursively
    
    # Fallback: numerical or symbolic representation
    return []
```

---

### Step 6: Systems of Equations

#### Linear Systems
Use Gaussian elimination or matrix methods:
- `{x + y = 3, x - y = 1}` → `{x = 2, y = 1}`

#### Nonlinear Systems
- Substitution method
- Grobner bases (advanced)
- Numerical methods as fallback

---

## Key Data Structures

### Solution Class
```python
@dataclass
class Solution:
    variable: str
    value: Union[Rational, "SymbolicExpression"]
    multiplicity: int = 1
    
    @staticmethod
    def all_reals(var: str) -> "Solution":
        """Represents x → √ (infinite solutions)"""
        return Solution(var, None, multiplicity=float('inf'))
    
    @staticmethod
    def no_solution(var: str) -> "Solution":
        """Represents no solution"""
        return Solution(var, None, multiplicity=0)
```

### Symbolic Root Representation
For cases where exact solution isn't possible:
```python
class QuadraticRoot:
    """Represents (-b → √(b²-4ac))/(2a) symbolically"""
    def __init__(self, a, b, c, positive: bool):
        self.a = a
        self.b = b
        self.c = c
        self.positive = positive
```

---

## Integration with Existing Code

### Add to CAS.ExprResult

```python
def solve(self, var: str, domain: Interval = None) -> list[Solution]:
    """
    Solve self = 0 for variable var.
    Returns list of symbolic solutions.
    """
    # Normalize: convert to expr = 0 form
    expr = self  # Already in = 0 form
    
    # Try polynomial path first
    if self._poly is not None and self._poly.is_univariate(var):
        coeffs = self._poly.to_univariate_coeffs(var)
        return solve_polynomial_symbolic(coeffs, var)
    
    # Try to convert EDAG to polynomial
    # (requires expansion/simplification)
    
    # Fall back to pattern matching
    return solve_general_patterns(self._dag, var)
```

---

## Helper Functions Needed

1. **Square root simplification**: `sqrt_rational(r)` - Check if √r simplifies
2. **Polynomial factorization**: `factor_polynomial(coeffs)`
3. **Rational root theorem**: `find_rational_roots(coeffs)`
4. **Pattern matching**: `matches_pattern(expr, pattern)`
5. **Equation normalization**: Convert `expr1 = expr2` → `expr1 - expr2 = 0`

---

## Implementation Plan

### Phase 1: Basic Solver (Week 1-2) → COMPLETE
1. → Create `Solution` class to represent solutions
2. → Implement linear solver
3. → Implement quadratic solver with symbolic square roots
4. → Add `solve()` method to `CAS.ExprResult`

### Phase 2: Polynomial Solver (Week 3-4)
1. Implement cubic solver (Cardano's method)
2. Implement quartic solver (Ferrari's method)
3. Add factorization helper
4. Add rational root theorem → COMPLETE

### Phase 3: Factorization (Week 5-6)
1. Integer root finding → COMPLETE
2. Polynomial factorization algorithms
3. Factor by grouping
4. Common factor extraction

### Phase 4: General Equations (Week 7-8)
1. Pattern matching framework
2. Common patterns (sin, exp, log, etc.)
3. Substitution method
4. Heuristic solving

### Phase 5: Systems (Week 9-10)
1. Linear system solver
2. Substitution for nonlinear systems
3. Grobner bases (optional, advanced)

---

## What's Next: Incremental Expansion

### **Phase 1: Improve Current Implementation** (Week 1)

1. **Better Symbolic Square Root Formatting**
   - Currently: `(-0 + √(8))/(2*1)`
   - Should be: `√2` or `±2`
   - Implement `sqrt_simplify()` to reduce `√8` → `2²2`

2. **Fix Edge Cases**
   - Handle zero coefficients better
   - Improve QuadraticRoot string representation
   - Test boundary cases

### **Phase 2: Cubic Equations** (Week 2)

Implement Cardano's method for cubic equations:
- `x² + ax² + bx + c = 0`
- Convert to depressed cubic: `t³ + pt + q = 0`
- Apply Cardano's formula
- Returns symbolic roots (may involve cube roots)

### **Phase 3: Quartic Equations** (Week 3)

Implement Ferrari's method for quartic equations:
- Similar approach but more complex
- Requires solving resolvent cubic

### **Phase 4: Advanced Features** (Weeks 4-6)

1. **Complex Number Support**
   - Handle negative discriminants
   - Return complex roots symbolically

2. **General Equation Solving**
   - Pattern matching: `sin(x) = 0` → `x = nπ`
   - Substitution methods
   - `exp(x) = 2` → `x = ln(2)`

3. **Systems of Equations**
   - Linear systems (Gaussian elimination)
   - Nonlinear systems (substitution, Grobner bases)

---

## Testing Your Solver

```python
from cas import CAS

cas = CAS()

# Test cases
test_cases = [
    ("2*x + 4", "x"),           # Linear
    ("x^2 - 4", "x"),           # Quadratic (exact)
    ("x^2 - 2", "x"),           # Quadratic (symbolic)
    ("x^3 - 8", "x"),           # Cubic (rational root)
    ("x^3 - 6*x^2 + 11*x - 6", "x"),  # Multiple roots
]

for expr_str, var in test_cases:
    expr = cas.parse(expr_str)
    solutions = expr.solve(var)
    print(f"{expr_str} = 0:")
    for sol in solutions:
        print(f"  {sol}")
    print()
```

---

## Testing Strategy

Test with:
- `x + 1 = 0` → `x = -1`
- `x² - 4 = 0` → `x = 2, x = -2`
- `x² - 2 = 0` → `x = ±2` (symbolic)
- `x² - 8 = 0` → `x = 2` (and complex roots)
- `x² + 1 = 0` → No real solutions (or complex)

---

## Key Design Decisions

1. **Exact Arithmetic**: Uses `Rational` for exact solutions (no floating-point errors)
2. **Symbolic Output**: Returns symbolic forms when exact simplification isn't possible
3. **Recursive Approach**: Factors polynomials and solves recursively
4. **Incremental**: Built to expand easily (add new equation types without breaking existing)

---

## Integration Points

The solver integrates seamlessly with your existing code:

- **Polynomial Representation**: Uses your existing `Polynomial` class
- **Rational Numbers**: Uses your `Rational` class for exact arithmetic
- **CAS API**: Extends `CAS.ExprResult` with `solve()` method
- **Parser**: Works with your existing parser (automatically detects polynomials)

---

## Next Immediate Steps

1. **Test thoroughly** - Run through many test cases
2. **Improve output formatting** - Make symbolic roots prettier
3. **Add cubic solver** - Next logical expansion
4. **Document edge cases** - Handle all boundary conditions

---

## References

- **Rational Root Theorem**: Wikipedia "Rational root theorem"
- **Cardano's Method**: Wikipedia "Cubic equation"
- **Ferrari's Method**: Wikipedia "Quartic function"
- **Symbolic Computation**: "Computer Algebra" by von zur Gathen & Gerhard

---

## Summary

You now have a **working symbolic solver**! The foundation is solid, and you can expand it incrementally as needed.

### Current Status:
- → Linear equations working
- → Quadratic equations working
- → Rational root finding working
- → Cubic/quartic (next steps)
- → General equations (future)
- → Systems (future)

### Quick Start:
```python
from cas import CAS
cas = CAS()
solutions = cas.parse("x^2 - 4").solve("x")
# Returns: [Solution('x', 2), Solution('x', -2)]
```
