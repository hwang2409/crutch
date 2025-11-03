from __future__ import annotations
from typing import Dict, List, Tuple, Iterable
from dataclasses import dataclass, field
from monomial import Monomial
from rational import Rational

@dataclass
class Polynomial:
	terms: List[Monomial] = field(default_factory=list)
	
	def __post_init__(self):
		self.normalize()
	
	@staticmethod
	def from_monomials(monoms: Iterable[Monomial]) -> 'Polynomial':
		p = Polynomial([])
		p.terms = list(monoms)
		p.normalize()
		return p
	
	def normalize(self) -> None:
		# combine like terms
		acc: Dict[Tuple[Tuple[str,int],...], Rational] = {}
		for m in self.terms:
			vm = tuple(sorted(m.vmap().items()))
			acc[vm] = acc.get(vm, Rational(0,1)) + m.coefficient()
		new_terms: List[Monomial] = []
		for vm, c in acc.items():
			if not c.is_zero():
				new_terms.append(Monomial(c, dict(vm)))
		new_terms.sort()
		self.terms = new_terms
	
	def is_zero(self) -> bool:
		return len(self.terms) == 0
	
	def is_constant(self) -> bool:
		return all(m.is_constant() for m in self.terms)
	
	def degree(self) -> int:
		return max((m.degree() for m in self.terms), default=0)
	
	def degree_var(self, var: str) -> int:
		deg = -1
		for m in self.terms:
			deg = max(deg, m.degree_var(var))
		return deg
	
	def __add__(self, rhs: 'Polynomial') -> 'Polynomial':
		return Polynomial.from_monomials(self.terms + rhs.terms)
	def __sub__(self, rhs: 'Polynomial') -> 'Polynomial':
		return Polynomial.from_monomials(self.terms + [ -Monomial(m.coefficient(), m.vmap()) for m in rhs.terms ])
	def __mul__(self, rhs: 'Polynomial') -> 'Polynomial':
		prods: List[Monomial] = []
		for a in self.terms:
			for b in rhs.terms:
				prods.append(a.mul_m(b))
		return Polynomial.from_monomials(prods)
	
	def scalar_mul(self, r: Rational) -> 'Polynomial':
		return Polynomial.from_monomials([ m.mul_r(r) for m in self.terms ])
	
	def pow(self, exp: int) -> 'Polynomial':
		if exp == 0:
			return Polynomial([Monomial(Rational(1,1), {})])
		res = Polynomial(self.terms)
		for _ in range(exp-1):
			res = res * self
		return res
	
	def eval(self, env: Dict[str, Rational]) -> Rational:
		total = Rational(0,1)
		for m in self.terms:
			r = m.coefficient()
			for var, exp in m.vmap().items():
				if var not in env:
					raise KeyError(f"Variable '{var}' not in env")
				r = r * (env[var] ** exp)
			total = total + r
		return total
	
	def derivative(self, var: str) -> 'Polynomial':
		monoms: List[Monomial] = []
		for m in self.terms:
			e = m.degree_var(var)
			if e == 0:
				continue
			v = m.vmap()
			v[var] = e-1
			if v[var] == 0:
				v.pop(var, None)
			monoms.append(Monomial(m.coefficient() * Rational(e,1), v))
		return Polynomial.from_monomials(monoms)
	
	def integral(self, var: str) -> 'Polynomial':
		monoms: List[Monomial] = []
		for m in self.terms:
			e = m.degree_var(var)
			v = m.vmap()
			v[var] = e+1
			monoms.append(Monomial(m.coefficient() / Rational(e+1,1), v))
		return Polynomial.from_monomials(monoms)
	
	def to_string(self) -> str:
		if len(self.terms) == 0:
			return "0"
		parts: List[str] = []
		for idx, m in enumerate(self.terms):
			s = m.to_string()
			if s.startswith('-'):
				body = s[1:]
				if idx == 0:
					parts.append(f"-{body}")
				else:
					parts.append(f" - {body}")
			else:
				if idx == 0:
					parts.append(s)
				else:
					parts.append(f" + {s}")
		return "".join(parts)
	def __str__(self) -> str:
		return self.to_string()
