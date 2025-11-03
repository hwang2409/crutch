from __future__ import annotations
from typing import List, Tuple
from rational import Rational
from monomial import Monomial
from polynomial import Polynomial

# Simple lexer + shunting-yard for polynomials
class Tok:
	def __init__(self, kind: str, lex: str = "", num: Rational | None = None):
		self.kind, self.lex, self.num = kind, lex, num

def tokenize(expr: str) -> List[Tok]:
	s = expr
	i, n = 0, len(s)
	toks: List[Tok] = []
	prev = None
	while i < n:
		c = s[i]
		if c.isspace():
			i += 1; continue
		if c in "+-*/^()":
			k = c
			i += 1
			# unary minus
			if k == '-' and (prev is None or prev.kind in ('+','-','*','/','^','(')):
				k = 'NEG'
			# implicit multiplication when an opening parenthesis follows an operand, e.g., "2(x+y)" or ")(" cases
			if k == '(' and prev and prev.kind in ('ID','NUM',')'):
				# insert an explicit '*'
				toks.append(Tok('*', '*'))
			t = Tok(k, k)
			toks.append(t)
			prev = t; continue
		# number (int or a/b or decimal)
		if c.isdigit():
			j = i
			has_dot = False
			while j < n and (s[j].isdigit() or (s[j]=='.' and not has_dot)):
				has_dot = has_dot or s[j]=='.'
				j += 1
			num_str = s[i:j]
			if has_dot:
				# implicit multiplication before a number, e.g., "x2" or ")3"
				if prev and prev.kind in ('ID','NUM',')'):
					toks.append(Tok('*', '*'))
				toks.append(Tok('NUM', num_str, Rational(int(float(num_str)*10**6), 10**6)))
				i = j; prev = toks[-1]; continue
			if j < n and s[j] == '/':
				j2 = j+1
				k2 = j2
				while k2 < n and s[k2].isdigit():
					k2 += 1
				frac = Rational(int(s[i:j]), int(s[j2:k2]))
				if prev and prev.kind in ('ID','NUM',')'):
					toks.append(Tok('*', '*'))
				toks.append(Tok('NUM', s[i:k2], frac))
				i = k2; prev = toks[-1]; continue
			else:
				if prev and prev.kind in ('ID','NUM',')'):
					toks.append(Tok('*', '*'))
				toks.append(Tok('NUM', num_str, Rational(int(num_str),1)))
				i = j; prev = toks[-1]; continue
		# identifier
		if c.isalpha() or c == '_':
			j = i+1
			while j < n and (s[j].isalnum() or s[j]=='_'):
				j += 1
			name = s[i:j]
			# If the identifier is purely alphabetic and multi-letter (e.g., "xy"),
			# split into single-letter variables with implicit multiplication.
			if name.isalpha() and len(name) > 1:
				for idx,ch in enumerate(name):
					if prev and prev.kind in ('ID','NUM',')'):
						toks.append(Tok('*', '*'))
					toks.append(Tok('ID', ch))
					prev = toks[-1]
				i = j; continue
			# implicit multiplication before a (single or complex) identifier, e.g., "2x" or ")x"
			if prev and prev.kind in ('ID','NUM',')'):
				toks.append(Tok('*', '*'))
			toks.append(Tok('ID', name))
			i = j; prev = toks[-1]; continue
		raise ValueError(f"Unexpected char {c}")
	return toks

prec = {'NEG':4,'^':3,'*':2,'/':2,'+':1,'-':1}
right_assoc = {'NEG', '^'}

def to_rpn(toks: List[Tok]) -> List[Tok]:
	out: List[Tok] = []
	op: List[Tok] = []
	for t in toks:
		if t.kind in ('NUM','ID'):
			out.append(t)
		elif t.kind in prec:
			while op and op[-1].kind != '(' and ((t.kind in right_assoc and prec[t.kind] < prec[op[-1].kind]) or (t.kind not in right_assoc and prec[t.kind] <= prec[op[-1].kind])):
				out.append(op.pop())
			op.append(t)
		elif t.kind == '(':
			op.append(t)
		elif t.kind == ')':
			while op and op[-1].kind != '(':
				out.append(op.pop())
			if not op: raise ValueError("Mismatched parens")
			op.pop()
		else:
			raise ValueError("Unknown token kind")
	while op:
		if op[-1].kind == '(': raise ValueError("Mismatched parens")
		out.append(op.pop())
	return out


def eval_rpn(rpn: List[Tok]) -> Polynomial:
	stack: List[Polynomial] = []
	for t in rpn:
		if t.kind == 'NUM':
			stack.append(Polynomial([Monomial(t.num, {})]))
		elif t.kind == 'ID':
			stack.append(Polynomial([Monomial(Rational(1,1), {t.lex:1})]))
		elif t.kind == 'NEG':
			if not stack: raise ValueError("neg missing operand")
			p = stack.pop()
			stack.append(p.scalar_mul(Rational(-1,1)))
		elif t.kind in ('+','-','*','/','^'):
			if len(stack) < 2: raise ValueError("binary op missing operands")
			b = stack.pop(); a = stack.pop()
			if t.kind == '+': stack.append(a + b)
			elif t.kind == '-': stack.append(a - b)
			elif t.kind == '*': stack.append(a * b)
			elif t.kind == '/':
				# only allow division by constant
				if not b.is_constant(): raise ValueError("Division by non-constant not supported in polynomial parser")
				# divide each term
				den = b.eval({})
				stack.append(a.scalar_mul(Rational(den.numerator(), den.denominator()) ** -1))
			elif t.kind == '^':
				if not b.is_constant(): raise ValueError("Exponent must be constant")
				exp = b.eval({})
				if not exp.is_int() or exp.numerator() < 0: raise ValueError("Exponent must be non-negative integer")
				stack.append(a.pow(exp.to_int()))
		else:
			raise ValueError("Unknown RPN token")
	if len(stack) != 1: raise ValueError("Invalid expression")
	return stack[-1]


def parse_polynomial(expr: str) -> Polynomial:
	toks = tokenize(expr)
	rpn = to_rpn(toks)
	return eval_rpn(rpn)
