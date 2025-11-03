from __future__ import annotations
from fractions import Fraction

class Rational:
	__slots__ = ("_f",)
	def __init__(self, num: int | Fraction, den: int | None = None) -> None:
		if isinstance(num, Fraction):
			self._f = num
		else:
			self._f = Fraction(num, 1 if den is None else den)
	def __add__(self, other: Rational) -> Rational:
		return Rational(self._f + other._f)
	def __sub__(self, other: Rational) -> Rational:
		return Rational(self._f - other._f)
	def __mul__(self, other: Rational | int) -> Rational:
		if isinstance(other, Rational):
			return Rational(self._f * other._f)
		return Rational(self._f * other)
	def __truediv__(self, other: Rational | int) -> Rational:
		if isinstance(other, Rational):
			if other._f == 0:
				raise ZeroDivisionError("division by zero")
			return Rational(self._f / other._f)
		if other == 0:
			raise ZeroDivisionError("division by zero")
		return Rational(self._f / other)
	def __neg__(self) -> Rational:
		return Rational(-self._f)
	def __pow__(self, exp: int) -> Rational:
		if exp == 0:
			return Rational(1,1)
		return Rational(self._f ** exp)
	def __eq__(self, other: object) -> bool:
		if not isinstance(other, Rational):
			return False
		return self._f == other._f
	def __lt__(self, other: Rational) -> bool:
		return self._f < other._f
	def __le__(self, other: Rational) -> bool:
		return self._f <= other._f
	def __gt__(self, other: Rational) -> bool:
		return self._f > other._f
	def __ge__(self, other: Rational) -> bool:
		return self._f >= other._f
	def is_zero(self) -> bool:
		return self._f == 0
	def is_int(self) -> bool:
		return self._f.denominator == 1
	def to_int(self) -> int:
		return self._f.numerator // self._f.denominator
	def numerator(self) -> int:
		return self._f.numerator
	def denominator(self) -> int:
		return self._f.denominator
	def to_string(self) -> str:
		if self._f.denominator == 1:
			return str(self._f.numerator)
		return f"{self._f.numerator}/{self._f.denominator}"
	def __str__(self) -> str:
		return self.to_string()
