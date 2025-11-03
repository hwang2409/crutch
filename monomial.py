from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
from rational import Rational

@dataclass(order=False)
class Monomial:
	coeff: Rational = field(default_factory=lambda: Rational(0,1))
	vars: Dict[str,int] = field(default_factory=dict)
	def coefficient(self) -> Rational:
		return self.coeff
	def degree(self) -> int:
		return sum(self.vars.values())
	def degree_var(self, var: str) -> int:
		return self.vars.get(var, 0)
	def is_constant(self) -> bool:
		return self.degree() == 0
	def is_zero(self) -> bool:
		return self.coeff.is_zero()
	def vmap(self) -> Dict[str,int]:
		return dict(self.vars)
	def variables(self) -> List[str]:
		return list(self.vars.keys())
	def mul_r(self, r: Rational) -> Monomial:
		return Monomial(self.coeff * r, dict(self.vars))
	def mul_m(self, other: Monomial) -> Monomial:
		v = dict(self.vars)
		for k,e in other.vars.items():
			v[k] = v.get(k,0) + e
		return Monomial(self.coeff * other.coeff, v)
	def pow(self, exp: int) -> Monomial:
		v = {k: e*exp for k,e in self.vars.items()}
		return Monomial(self.coeff ** exp, v)
	def __neg__(self) -> Monomial:
		return Monomial(-self.coeff, dict(self.vars))
	def is_like_term(self, other: Monomial) -> bool:
		# compare variable map only
		return self._unit().vars == other._unit().vars
	def _unit(self) -> Monomial:
		return Monomial(Rational(1,1) if not self.is_zero() else Rational(0,1), dict(self.vars))
	def sub(self, var: str, value: Rational) -> Monomial:
		e = self.vars.get(var, 0)
		if e == 0:
			return Monomial(self.coeff, dict(self.vars))
		v = dict(self.vars)
		del v[var]
		return Monomial(self.coeff * (value ** e), v)
	def __lt__(self, other: Monomial) -> bool:
		# grevlex
		a, b = self.degree(), other.degree()
		if a != b:
			return a < b
		# reverse lex by variable name ordering
		all_vars = sorted(set(self.vars) | set(other.vars))
		for var in reversed(all_vars):
			e1, e2 = self.vars.get(var, 0), other.vars.get(var, 0)
			if e1 != e2:
				return e1 > e2
		return self.coeff < other.coeff
	def to_string(self) -> str:
		# If no variables, print coefficient directly
		if len(self.vars) == 0:
			return self.coeff.to_string()
		# Determine sign and absolute coefficient
		sign = "-" if self.coeff < Rational(0,1) else ""
		abs_coeff = -self.coeff if sign == "-" else self.coeff
		coeff_part = "" if abs_coeff == Rational(1,1) else abs_coeff.to_string()
		# Order variables by name for stable printing
		var_names = sorted(k for k in self.vars.keys() if self.vars[k] != 0)
		vars_part = ""
		for name in var_names:
			exp = self.vars[name]
			if exp == 1:
				vars_part += f"{name}"
			else:
				vars_part += f"{name}^{exp}"
		return f"{sign}{coeff_part}{vars_part}"
	def __str__(self) -> str:
		return self.to_string()
