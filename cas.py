from __future__ import annotations
from typing import Dict, Any, Union
from polynomial_parser import parse_polynomial
from polynomial import Polynomial
from rational import Rational
from edag import EDAG

NumberLike = Union[int, float, Rational]

class CAS:
	def __init__(self) -> None:
		pass
	def parse(self, expr: str) -> EDAG:
		d = EDAG(); d.parse(expr); return d
	def eval(self, expr: str, env: Dict[str, NumberLike] | None = None) -> Any:
		# Try polynomial path first
		try:
			p = parse_polynomial(expr)
			env_r = {k: (v if isinstance(v, Rational) else Rational(v,1) if isinstance(v,int) else Rational(int(v*10**6),10**6)) for k,v in (env or {}).items()}
			return p.eval(env_r)
		except Exception:
			return EDAG().parse(expr)
	def differentiate(self, expr: str, var: str) -> str:
		p = parse_polynomial(expr)
		return str(p.derivative(var))
	def integrate(self, expr: str, var: str) -> str:
		p = parse_polynomial(expr)
		return str(p.integral(var))
	def simplify(self, expr: str) -> str:
		p = parse_polynomial(expr)
		p.normalize()
		return str(p)
