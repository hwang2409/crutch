## Current Status

**Expression Parser & Evaluator**
- Infix to postfix conversion with proper precedence/associativity
- Unary minus handling (`-x`, `-(x+y)`)
- N-ary addition and multiplication
- Built-in functions: `sin`, `cos`, `tan`, `log`, `exp`, `sqrt`, `abs`
- Constants: `pi`, `e`, `tau`
- Variable substitution and evaluation

## Development Roadmap

### Phase 1: Core Infrastructure
- [x] DAG-based expression representation
- [x] Parser with shunting-yard algorithm
- [x] Expression evaluation with variable substitution
- [x] Operator precedence and associativity
- [x] Unary operations and functions

### Phase 2: Pretty Printing & LaTeX
- [x] `to_string()` with minimal parentheses
- [x] `to_latex()` with proper mathematical formatting
- [x] Function rendering (`sin`, `cos`, `sqrt`, `abs`)
- [x] Fraction and power notation

### Phase 3: Simplification Engine
- [x] Basic algebraic identities
  - [x] `x + 0 → x`, `0 + x → x`
  - [x] `x * 1 → x`, `x * 0 → 0`
  - [x] `x - 0 → x`, `0 - x → -x`
  - [x] `x^1 → x`, `x^0 → 1` (x ≠ 0)
- [x] Function simplifications
  - [x] `sin(0) = 0`, `cos(0) = 1`
  - [x] `log(1) = 0`, `exp(0) = 1`
  - [x] `log(exp(x)) → x`
- [x] Constant folding
- [x] Associative flattening for `+/*`
- [x] Commutative sorting

### Phase 4: Exact Arithmetic
- [x] `Rational` class with GCD normalization
- [x] Double to rational conversion (continued fractions)
- [x] Exact arithmetic operations
- [x] Mixed exact/approximate evaluation
- [x] Overflow protection

### Phase 5: Polynomial Support
- [x] Monomial representation
- [ ] Sparse polynomial storage
- [x] Polynomial arithmetic (`+`, `-`, `*`)
- [x] Like term combining
- [x] Polynomial detection in DAG
- [x] Degree calculation
- [x] Evaluation and substitution

### Phase 6: Symbolic Differentiation
- [x] Basic differentiation rules
  - [x] `d/dx(x) = 1`, `d/dx(c) = 0`
  - [x] Linearity: `d/dx(af + bg) = a*f' + b*g'`
  - [x] Product rule: `d/dx(fg) = f'g + fg'`
  - [x] Quotient rule: `d/dx(f/g) = (f'g - fg')/g²`
  - [x] Chain rule: `d/dx(f(g)) = f'(g) * g'`
- [x] Function derivatives
  - [x] `sin' → cos`, `cos' → -sin`
  - [x] `tan' → sec²`, `log' → 1/x`
  - [x] `exp' → exp`, `sqrt' → 1/(2√x)`
- [x] Power rule: `d/dx(x^n) = n*x^(n-1)`
- [x] Automatic simplification of derivatives

### Phase 7: Advanced Features
- [ ] Substitution system
- [ ] Canonicalization
- [ ] GCD and factorization
- [ ] Univariate polynomial solving
- [ ] Complex number support
- [ ] Matrix operations
- [ ] Series expansion

### Phase 8: User Interface
- [ ] REPL/CLI interface
- [ ] Command parsing
- [ ] Error handling and diagnostics
- [ ] Interactive help system
- [ ] Batch processing

# Testing Strategy
### Core Functionality Tests
- [ ] Parsing precedence: `2^3^2`, `a-b-c`, `-x`, `-(x+y)`
- [ ] Evaluation accuracy with various inputs
- [ ] Error handling for invalid expressions
- [ ] Round-trip: `parse → to_string → parse`

### Simplification Tests
- [ ] Identity rules: `x+0→x`, `x*1→x`, `x*0→0`
- [ ] Function simplifications: `sin(0)→0`, `log(exp(x))→x`
- [ ] Like term combining: `2*x+3*x→5*x`

### Differentiation Tests
- [ ] Basic rules: `d/dx(x^n)`, `d/dx(sin(x))`, `d/dx(log(x))`
- [ ] Chain rule: `d/dx(sin(x^2))`
- [ ] Product rule: `d/dx(x*sin(x))`
- [ ] Numerical validation against finite differences

### Performance Tests
- [ ] Large expression parsing
- [ ] Deep nesting evaluation
- [ ] Memory usage optimization
- [ ] Evaluation speed benchmarks
