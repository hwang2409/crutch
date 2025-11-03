from __future__ import annotations
from typing import Dict, Any, Union
from parser import parse_polynomial, parse_expression_edag
from polynomial import Polynomial
from rational import Rational
from edag import EDAG
import re

NumberLike = Union[int, float, Rational]

class CAS:
	def __init__(self) -> None:
		pass
	def parse(self, expr: str) -> EDAG:
		d = EDAG(); d.parse(expr); return d
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
	def eval(self, expr: str, env: Dict[str, NumberLike] | None = None) -> Any:
		# If expression contains a function-call pattern like 'name(...)', use EDAG-based evaluator
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
	def differentiate(self, expr: str, var: str) -> 'CAS.ExprResult':
		# Prefer EDAG if we detect function calls or decimals
		if re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\(", expr) or re.search(r"\d+\.\d+", expr):
			dag = parse_expression_edag(expr)
			ddag = dag.derivative(var)
			return CAS.ExprResult(dag=ddag)
		# Try polynomial path; if it fails (e.g., e^x, x^(1/2)), fall back to EDAG
		try:
			p = parse_polynomial(expr)
			return CAS.ExprResult(poly=p.derivative(var))
		except Exception:
			dag = parse_expression_edag(expr)
			ddag = dag.derivative(var)
			return CAS.ExprResult(dag=ddag)
	def integrate(self, expr: str, var: str) -> str:
		p = parse_polynomial(expr)
		return str(p.integral(var))
	def simplify(self, expr: str) -> str:
		p = parse_polynomial(expr)
		p.normalize()
		return str(p)
