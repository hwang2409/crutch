from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterable, Union
import math

from rational import Rational
from monomial import Monomial
from polynomial import Polynomial
from edag import EDAG, Node

# =====================
# Polynomial-only parser
# =====================


class PolyTok:
    def __init__(self, kind: str, lex: str = "", num: Rational | None = None):
        self.kind, self.lex, self.num = kind, lex, num


def poly_tokenize(expr: str) -> List[PolyTok]:
    s = expr
    i, n = 0, len(s)
    toks: List[PolyTok] = []
    prev: Optional[PolyTok] = None
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c in "+-*/^()":
            k = c
            i += 1
            if k == "-" and (
                prev is None or prev.kind in ("+", "-", "*", "/", "^", "(")
            ):
                k = "NEG"
            # implicit multiplication when an opening parenthesis follows an operand
            if k == "(" and prev and prev.kind in ("ID", "NUM", ")"):
                toks.append(PolyTok("*", "*"))
            t = PolyTok(k, k)
            toks.append(t)
            prev = t
            continue
        if c.isdigit():
            j = i
            has_dot = False
            while j < n and (s[j].isdigit() or (s[j] == "." and not has_dot)):
                has_dot = has_dot or s[j] == "."
                j += 1
            num_str = s[i:j]
            if has_dot:
                if prev and prev.kind in ("ID", "NUM", ")"):
                    toks.append(PolyTok("*", "*"))
                toks.append(
                    PolyTok(
                        "NUM", num_str, Rational(int(float(num_str) * 10**6), 10**6)
                    )
                )
                i = j
                prev = toks[-1]
                continue
            if j < n and s[j] == "/" and (j + 1 < n and s[j + 1].isdigit()):
                j2 = j + 1
                k2 = j2
                while k2 < n and s[k2].isdigit():
                    k2 += 1
                frac = Rational(int(s[i:j]), int(s[j2:k2]))
                if prev and prev.kind in ("ID", "NUM", ")"):
                    toks.append(PolyTok("*", "*"))
                toks.append(PolyTok("NUM", s[i:k2], frac))
                i = k2
                prev = toks[-1]
                continue
            else:
                if prev and prev.kind in ("ID", "NUM", ")"):
                    toks.append(PolyTok("*", "*"))
                toks.append(PolyTok("NUM", num_str, Rational(int(num_str), 1)))
                i = j
                prev = toks[-1]
                continue
        if c.isalpha() or c == "_":
            j = i + 1
            while j < n and (s[j].isalnum() or s[j] == "_"):
                j += 1
            name = s[i:j]
            lname = name.lower()
            if lname in ("pi", "e"):
                if prev and prev.kind in ("ID", "NUM", ")"):
                    toks.append(PolyTok("*", "*"))
                val = 3.141592653589793 if lname == "pi" else 2.718281828459045
                toks.append(PolyTok("NUM", name, Rational(int(val * 10**6), 10**6)))
                i = j
                prev = toks[-1]
                continue
            if name.isalpha() and len(name) > 1:
                for ch in name:
                    if prev and prev.kind in ("ID", "NUM", ")"):
                        toks.append(PolyTok("*", "*"))
                    toks.append(PolyTok("ID", ch))
                    prev = toks[-1]
                i = j
                continue
            if prev and prev.kind in ("ID", "NUM", ")"):
                toks.append(PolyTok("*", "*"))
            toks.append(PolyTok("ID", name))
            i = j
            prev = toks[-1]
            continue
        raise ValueError(f"Unexpected char {c}")
    return toks


_poly_prec = {"NEG": 4, "^": 3, "*": 2, "/": 2, "+": 1, "-": 1}
_poly_right_assoc = {"NEG", "^"}


def poly_to_rpn(toks: List[PolyTok]) -> List[PolyTok]:
    out: List[PolyTok] = []
    op: List[PolyTok] = []
    for t in toks:
        if t.kind in ("NUM", "ID"):
            out.append(t)
        elif t.kind in _poly_prec:
            while (
                op
                and op[-1].kind != "("
                and (
                    (
                        t.kind in _poly_right_assoc
                        and _poly_prec[t.kind] < _poly_prec[op[-1].kind]
                    )
                    or (
                        t.kind not in _poly_right_assoc
                        and _poly_prec[t.kind] <= _poly_prec[op[-1].kind]
                    )
                )
            ):
                out.append(op.pop())
            op.append(t)
        elif t.kind == "(":
            op.append(t)
        elif t.kind == ")":
            while op and op[-1].kind != "(":
                out.append(op.pop())
            if not op:
                raise ValueError("Mismatched parens")
            op.pop()
        else:
            raise ValueError("Unknown token kind")
    while op:
        if op[-1].kind == "(":
            raise ValueError("Mismatched parens")
        out.append(op.pop())
    return out


def poly_eval_rpn(rpn: List[PolyTok]) -> Polynomial:
    stack: List[Polynomial] = []
    for t in rpn:
        if t.kind == "NUM":
            stack.append(Polynomial([Monomial(t.num, {})]))
        elif t.kind == "ID":
            stack.append(Polynomial([Monomial(Rational(1, 1), {t.lex: 1})]))
        elif t.kind == "NEG":
            if not stack:
                raise ValueError("neg missing operand")
            p = stack.pop()
            stack.append(p.scalar_mul(Rational(-1, 1)))
        elif t.kind in ("+", "-", "*", "/", "^"):
            if len(stack) < 2:
                raise ValueError("binary op missing operands")
            b = stack.pop()
            a = stack.pop()
            if t.kind == "+":
                stack.append(a + b)
            elif t.kind == "-":
                stack.append(a - b)
            elif t.kind == "*":
                stack.append(a * b)
            elif t.kind == "/":
                if not b.is_constant():
                    raise ValueError(
                        "Division by non-constant not supported in polynomial parser"
                    )
                den = b.eval({})
                stack.append(
                    a.scalar_mul(Rational(den.numerator(), den.denominator()) ** -1)
                )
            elif t.kind == "^":
                if not b.is_constant():
                    raise ValueError("Exponent must be constant")
                exp = b.eval({})
                if not exp.is_int() or exp.numerator() < 0:
                    raise ValueError("Exponent must be non-negative integer")
                stack.append(a.pow(exp.to_int()))
        else:
            raise ValueError("Unknown RPN token")
    if len(stack) != 1:
        raise ValueError("Invalid expression")
    return stack[-1]


def parse_polynomial(expr: str) -> Polynomial:
    toks = poly_tokenize(expr)
    rpn = poly_to_rpn(toks)
    return poly_eval_rpn(rpn)


# =====================
# General expression parser (non-polynomial)
# =====================


@dataclass
class Expr:
    pass


@dataclass
class Number(Expr):
    value: Rational


@dataclass
class Variable(Expr):
    name: str


@dataclass
class UnaryOp(Expr):
    op: str
    arg: Expr


@dataclass
class BinaryOp(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass
class FunctionCall(Expr):
    name: str
    args: List[Expr]


class ExprTok:
    def __init__(self, kind: str, lex: str = "", num: Optional[Rational] = None):
        self.kind, self.lex, self.num = kind, lex, num


def expr_tokenize(expr: str) -> List[ExprTok]:
    s = expr
    i, n = 0, len(s)
    toks: List[ExprTok] = []
    prev: Optional[ExprTok] = None
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c in "+-*/^(),":
            k = c
            i += 1
            if k == "-" and (
                prev is None or prev.kind in ("+", "-", "*", "/", "^", "(", ",")
            ):
                k = "NEG"
            if k == "(" and prev and prev.kind in ("ID", "NUM", ")"):
                toks.append(ExprTok("*", "*"))
            t = ExprTok(k, k)
            toks.append(t)
            prev = t
            continue
        if c.isdigit():
            j = i
            has_dot = False
            while j < n and (s[j].isdigit() or (s[j] == "." and not has_dot)):
                has_dot = has_dot or s[j] == "."
                j += 1
            num_str = s[i:j]
            if has_dot:
                if prev and prev.kind in ("ID", "NUM", ")"):
                    toks.append(ExprTok("*", "*"))
                toks.append(
                    ExprTok(
                        "NUM", num_str, Rational(int(float(num_str) * 10**6), 10**6)
                    )
                )
                i = j
                prev = toks[-1]
                continue
            if j < n and s[j] == "/" and (j + 1 < n and s[j + 1].isdigit()):
                j2 = j + 1
                k2 = j2
                while k2 < n and s[k2].isdigit():
                    k2 += 1
                frac = Rational(int(s[i:j]), int(s[j2:k2]))
                if prev and prev.kind in ("ID", "NUM", ")"):
                    toks.append(ExprTok("*", "*"))
                toks.append(ExprTok("NUM", s[i:k2], frac))
                i = k2
                prev = toks[-1]
                continue
            else:
                if prev and prev.kind in ("ID", "NUM", ")"):
                    toks.append(ExprTok("*", "*"))
                toks.append(ExprTok("NUM", num_str, Rational(int(num_str), 1)))
                i = j
                prev = toks[-1]
                continue
        if c.isalpha() or c == "_":
            j = i + 1
            while j < n and (s[j].isalnum() or s[j] == "_"):
                j += 1
            name = s[i:j]
            lname = name.lower()
            if lname in ("pi", "e"):
                if prev and prev.kind in ("ID", "NUM", ")"):
                    toks.append(ExprTok("*", "*"))
                val = 3.141592653589793 if lname == "pi" else 2.718281828459045
                toks.append(ExprTok("NUM", name, Rational(int(val * 10**6), 10**6)))
                i = j
                prev = toks[-1]
                continue
            k = j
            while k < n and s[k].isspace():
                k += 1
            is_func = k < n and s[k] == "("
            if prev and prev.kind in ("ID", "NUM", ")"):
                toks.append(ExprTok("*", "*"))
            toks.append(ExprTok("FUNC" if is_func else "ID", name))
            i = j
            prev = toks[-1]
            continue
        raise ValueError(f"Unexpected char {c}")
    return toks


_expr_prec = {"NEG": 5, "^": 4, "*": 3, "/": 3, "+": 2, "-": 2, ",": 1}
_expr_right_assoc = {"NEG", "^"}


def expr_to_rpn(toks: List[ExprTok]) -> List[ExprTok]:
    out: List[ExprTok] = []
    op: List[ExprTok] = []
    func_stack: List[str] = []
    for t in toks:
        if t.kind in ("NUM", "ID"):
            out.append(t)
        elif t.kind == "FUNC":
            op.append(t)
            func_stack.append(t.lex)
        elif t.kind == ",":
            while op and op[-1].kind != "(":
                out.append(op.pop())
        elif t.kind in _expr_prec:
            while (
                op
                and op[-1].kind != "("
                and (
                    (
                        t.kind in _expr_right_assoc
                        and _expr_prec[t.kind] < _expr_prec[op[-1].kind]
                    )
                    or (
                        t.kind not in _expr_right_assoc
                        and _expr_prec[t.kind] <= _expr_prec[op[-1].kind]
                    )
                )
            ):
                out.append(op.pop())
            op.append(t)
        elif t.kind == "(":
            op.append(t)
        elif t.kind == ")":
            while op and op[-1].kind != "(":
                out.append(op.pop())
            if not op:
                raise ValueError("Mismatched parens")
            op.pop()
            if op and op[-1].kind == "FUNC":
                out.append(op.pop())
                if func_stack:
                    func_stack.pop()
        else:
            raise ValueError("Unknown token kind")
    while op:
        if op[-1].kind == "(":
            raise ValueError("Mismatched parens")
        out.append(op.pop())
    return out


def rpn_to_ast(rpn: List[ExprTok]):
    stack: List[Any] = []
    for t in rpn:
        if t.kind == "NUM":
            stack.append(Number(t.num if t.num is not None else Rational(0, 1)))
        elif t.kind == "ID":
            stack.append(Variable(t.lex))
        elif t.kind == "NEG":
            if not stack:
                raise ValueError("neg missing operand")
            stack.append(UnaryOp("-", stack.pop()))
        elif t.kind in ("+", "-", "*", "/", "^"):
            if len(stack) < 2:
                raise ValueError("binary op missing operands")
            b = stack.pop()
            a = stack.pop()
            stack.append(BinaryOp(t.lex, a, b))
        elif t.kind == "FUNC":
            if not stack:
                raise ValueError("function missing argument")
            arg = stack.pop()
            stack.append(FunctionCall(t.lex, [arg]))
        else:
            raise ValueError("Unknown RPN token")
    if len(stack) != 1:
        raise ValueError("Invalid expression")
    return stack[-1]


def parse_expression(expr: str):
    toks = expr_tokenize(expr)
    rpn = expr_to_rpn(toks)
    return rpn_to_ast(rpn)


def rpn_to_edag(rpn: List[ExprTok]) -> EDAG:
    dag = EDAG()
    stack: List[str] = []

    def add_node(node: Node) -> str:
        nid = dag._nid()
        dag.g.add_node(nid, data=node)
        return nid

    for t in rpn:
        if t.kind == "NUM":
            nid = add_node(
                Node(
                    "CONST",
                    t.lex,
                    value=(t.num if t.num is not None else Rational(0, 1)),
                )
            )
            stack.append(nid)
        elif t.kind == "ID":
            nid = add_node(Node("VAR", t.lex))
            stack.append(nid)
        elif t.kind == "NEG":
            if not stack:
                raise ValueError("neg missing operand")
            a = stack.pop()
            node = Node("OP", "-", op="-", is_unary=True, children=[a])
            nid = add_node(node)
            dag.g.add_edge(a, nid)
            stack.append(nid)
        elif t.kind in ("+", "-", "*", "/", "^"):
            if len(stack) < 2:
                raise ValueError("binary op missing operands")
            b = stack.pop()
            a = stack.pop()
            node = Node("OP", t.lex, op=t.lex, is_unary=False, children=[a, b])
            nid = add_node(node)
            dag.g.add_edge(a, nid)
            dag.g.add_edge(b, nid)
            stack.append(nid)
        elif t.kind == "FUNC":
            if not stack:
                raise ValueError("function missing argument")
            a = stack.pop()
            node = Node("OP", t.lex, op=t.lex, is_unary=True, children=[a])
            nid = add_node(node)
            dag.g.add_edge(a, nid)
            stack.append(nid)
        else:
            raise ValueError("Unknown RPN token")
    if len(stack) != 1:
        raise ValueError("Invalid expression")
    dag.root = stack[-1]
    return dag


def parse_expression_edag(expr: str) -> EDAG:
    toks = expr_tokenize(expr)
    rpn = expr_to_rpn(toks)
    return rpn_to_edag(rpn)


_funcs: Dict[str, Callable[..., float]] = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "exp": math.exp,
    "log": math.log,
    "ln": math.log,
    "sqrt": math.sqrt,
}


def _to_float(value: Rational | int | float) -> float:
    if isinstance(value, Rational):
        return value.numerator() / value.denominator()
    if isinstance(value, int):
        return float(value)
    return value


def eval_expression(ast: Any, env: Optional[Dict[str, Any]] = None) -> Any:
    env = env or {}

    def _norm(v: Any) -> Any:
        if isinstance(v, int):
            return Rational(v, 1)
        return v

    def _bin(op: str, a: Any, b: Any) -> Any:
        if isinstance(a, float) or isinstance(b, float):
            fa, fb = _to_float(a), _to_float(b)
            if op == "+":
                return fa + fb
            if op == "-":
                return fa - fb
            if op == "*":
                return fa * fb
            if op == "/":
                return fa / fb
            if op == "^":
                return fa**fb
            raise ValueError(f"Unsupported binary op {op}")
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            return a / b
        if op == "^":
            if isinstance(b, int):
                return a**b
            if isinstance(b, Rational) and b.is_int():
                return a ** b.to_int()
            return _to_float(a) ** _to_float(b)
        raise ValueError(f"Unsupported binary op {op}")

    if isinstance(ast, Number):
        return ast.value
    if isinstance(ast, Variable):
        v = env.get(ast.name)
        if v is None:
            raise KeyError(f"Variable '{ast.name}' not in env")
        return _norm(v)
    if isinstance(ast, UnaryOp):
        val = eval_expression(ast.arg, env)
        if ast.op == "-":
            return -val if not isinstance(val, Rational) else -val
        raise ValueError(f"Unsupported unary op {ast.op}")
    if isinstance(ast, BinaryOp):
        a = eval_expression(ast.left, env)
        b = eval_expression(ast.right, env)
        return _bin(ast.op, a, b)
    if isinstance(ast, FunctionCall):
        fn = _funcs.get(ast.name)
        if fn is None:
            raise ValueError(f"Unknown function {ast.name}")
        args = [_to_float(eval_expression(arg, env)) for arg in ast.args]
        return fn(*args)
    raise ValueError("Unknown AST node")
